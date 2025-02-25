/*
 * dsaX_average_multitime.c
 *
 * This program reads from a PSRDADA ring buffer that was filled by dsaX_capture.c.
 * Unlike simpler examples where one DADA block holds one time sample, here each
 * DADA block contains many UDP packets that have been reassembled into a continuous
 * stream of time samples. Each complete time sample is stored as:
 *
 *   [12 antpols] x [3072 frequency channels] = 12 * 3072 bytes
 *
 * (Each voltage sample is stored as a packed 8-bit value: 4 bits for the real part
 * and 4 bits for the imaginary part, in two's complement format.)
 *
 * The program processes each time sample by:
 *   - Unpacking each byte to extract the real and imaginary parts.
 *   - Computing the intensity as (real^2 + imag^2).
 *   - Accumulating (summing) the intensity for INT_FACTOR consecutive time samples.
 *
 * Once INT_FACTOR time samples have been accumulated, it averages the result (sample by sample)
 * and writes the integrated data (averaged intensity as floats) to an output file.
 *
 * Usage:
 *    dsaX_average_multitime [-k dada_key] [-o output_filename]
 *
 * -k dada_key         : Key of the DADA ring buffer (default is DEFAULT_KEY)
 * -o output_filename  : Output file for averaged data (default "averaged.dat")
 *
 * Compile with the PSRDADA libraries and include paths.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "dada_hdu.h"
#include "ipcio.h"
#include "multilog.h"
#include "dada_def.h"
#include "ascii_header.h"

/* Dimensions for one complete time sample */
#define NANTPOL 12
#define NCHAN 512
#define TSAMPLE_SIZE (NANTPOL * NCHAN)  // bytes per time sample
#define DADA_HEADER_SIZE 4096


/* Integration (averaging) factor: number of time samples to average together */
#define INT_FACTOR 1

/* Default DADA ring buffer key (should match what dsaX_capture.c uses) */
#ifndef DEFAULT_KEY
#define DEFAULT_KEY 0x4000  /* adjust this value as needed */
#endif

/*
 * Helper function: sign-extend a 4-bit nibble (interpreted as two's complement)
 */
static inline int8_t sign_extend_4bit(uint8_t nibble)
{
    return (nibble & 0x8) ? (nibble - 16) : nibble;
}

int main(int argc, char **argv)
{
    key_t dada_key = DEFAULT_KEY;
    char * output_filename = "averaged.dat";
    int opt;

    /* Parse command-line options */
    while ((opt = getopt(argc, argv, "k:o:")) != -1)
    {
        switch (opt)
        {
            case 'k':
                dada_key = (key_t) strtoul(optarg, NULL, 0);
                break;
            case 'o':
                output_filename = optarg;
                break;
            default:
                fprintf(stderr, "Usage: %s [-k dada_key] [-o output_filename]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    /* Open a multilog for logging messages */
    multilog_t * log = multilog_open("dsaX_average_multitime", 0);

    /* Create and configure the DADA HDU for reading */
    dada_hdu_t * hdu_in = dada_hdu_create(log);
    dada_hdu_set_key(hdu_in, dada_key);
    if (dada_hdu_connect(hdu_in) < 0)
    {
        fprintf(stderr, "Could not connect to DADA ring buffer (key=0x%x)\n", dada_key);
        exit(EXIT_FAILURE);
    }
    if (dada_hdu_lock_read(hdu_in) < 0)
    {
        fprintf(stderr, "Could not lock DADA ring buffer for reading\n");
        exit(EXIT_FAILURE);
    }

    /* 
     * Each DADA block is much larger than one time sample.
     * Its size is given by ipcbuf_get_bufsz, which expects an ipcbuf_t*.
     * Since hdu_in->data_block is declared as an ipcio_t*, cast it to (ipcbuf_t*).
     */
    uint64_t block_size = ipcbuf_get_bufsz((ipcbuf_t*) hdu_in->data_block);
    fprintf(stderr, "Block size: %lu bytes\n", block_size);
    int num_time_samples = block_size / TSAMPLE_SIZE;
    if (num_time_samples <= 0)
    {
        fprintf(stderr, "Block size (%lu bytes) is too small for a single time sample (%d bytes)\n",
                block_size, TSAMPLE_SIZE);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Each DADA block contains %d time samples\n", TSAMPLE_SIZE);

    /* Open the output file for writing the averaged data */
    FILE * fout = fopen(output_filename, "wb");
    if (!fout)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    /* Allocate an accumulator buffer for integration.
       This buffer holds the sum of intensities for one integrated output,
       with one element per [antpol, channel] (i.e. TSAMPLE_SIZE elements).
       We use float to allow fractional averages.
    */
    float * accum = (float *) calloc(TSAMPLE_SIZE, sizeof(float));
    if (!accum)
    {
        perror("calloc");
        exit(EXIT_FAILURE);
    }
    int integration_count = 0;  // counts how many time samples have been accumulated

    unsigned long long total_time_samples_processed = 0;
    fprintf(stderr, "Starting integration: accumulating %d time samples per averaged output\n", INT_FACTOR);

    /* Main processing loop: keep reading DADA blocks */
    while (1)
    {
        uint64_t curbufsz = 0;
        uint64_t block_id = 0;
        /* Get next DADA block. Note that ipcio_open_block_read now requires additional parameters. */
        char * block = ipcio_open_block_read(hdu_in->data_block, &curbufsz, &block_id);
        if (!block)
        {
            fprintf(stderr, "ipcio_open_block_read failed\n");
            break;
        }

        /* Skip the DADA header */
        char * data_start = block + DADA_HEADER_SIZE;
        uint64_t data_size = curbufsz - DADA_HEADER_SIZE;

        /* Process each time sample within the block */
        for (int ts = 0; ts < num_time_samples; ts++)
        {
            uint8_t * tsample = (uint8_t *) data_start + ts * TSAMPLE_SIZE;
            /* For each byte in the time sample, unpack and compute intensity */
            for (int i = 0; i < TSAMPLE_SIZE; i++)
            {
                uint8_t packed = tsample[i];
                int8_t real = sign_extend_4bit(packed >> 4);
                int8_t imag = sign_extend_4bit(packed & 0x0F);
                int intensity = real * real + imag * imag;
                accum[i] += (float) intensity;
            }
            integration_count++;
            total_time_samples_processed++;

            /* When we have accumulated INT_FACTOR time samples, compute the average */
            if (integration_count == INT_FACTOR)
            {
                float * avg = (float *) malloc(TSAMPLE_SIZE * sizeof(float));
                if (!avg)
                {
                    perror("malloc");
                    exit(EXIT_FAILURE);
                }
                for (int i = 0; i < TSAMPLE_SIZE; i++)
                {
                    avg[i] = accum[i] / (float) INT_FACTOR;
                }
                size_t written = fwrite(avg, sizeof(float), TSAMPLE_SIZE, fout);
                if (written != (size_t) TSAMPLE_SIZE)
                {
                    fprintf(stderr, "Error writing averaged data to output file\n");
                }
                fflush(fout);
                fprintf(stderr, "Averaged %llu time samples\n", total_time_samples_processed);

                /* Reset accumulator and integration counter for next integration interval */
                memset(accum, 0, TSAMPLE_SIZE * sizeof(float));
                integration_count = 0;
                free(avg);
            }
        } // end loop over time samples in the block

        /* Now close the block; ipcio_close_block_read requires the number of bytes read */
        if (ipcio_close_block_read(hdu_in->data_block, curbufsz) < 0)
        {
            fprintf(stderr, "ipcio_close_block_read failed\n");
            break;
        }
    } // end main loop

    /* Cleanup */
    free(accum);
    fclose(fout);
    dada_hdu_unlock_read(hdu_in);
    dada_hdu_destroy(hdu_in);
    multilog_close(log);

    return 0;
}
