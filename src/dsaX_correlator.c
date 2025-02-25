/*
 * dsaX_correlator.c
 *
 * Example correlator that reads voltage data from a DADA buffer,
 * cross correlates (computes visibilities) from 12 antennas (single polarization),
 * and then averages those visibilities over a fixed integration time.
 *
 * The voltage data are assumed to be stored as 4+4-bit complex numbers in a single byte,
 * where the high nibble (bits 7-4) is the real part and the low nibble (bits 3-0) is the imaginary part.
 *
 * To compile, you will need to link with the DADA libraries.
 *
 * Usage:
 *   ./dsaX_correlator <DADA key in hex>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include "dada_hdu.h"
#include "ipcbuf.h"

#define NANTS 12                   // Number of antennas
#define INTEGRATION_SAMPLES 1024   // Number of time samples to average over

/* A simple structure for holding accumulated complex visibilities */
typedef struct {
    double re;
    double im;
} complex_d;

/*
 * decode_sample: Decode an 8-bit sample into its real and imaginary parts.
 * The sample is assumed to be stored as 4 bits real and 4 bits imaginary,
 * in two's complement form.
 */
static inline void decode_sample(uint8_t sample, int *re, int *im) {
    int r = (sample >> 4) & 0x0F;
    int i = sample & 0x0F;
    /* Convert 4-bit two's complement to signed integer */
    if (r & 0x08)
        r -= 16;
    if (i & 0x08)
        i -= 16;
    *re = r;
    *im = i;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <DADA key in hex>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    /* Parse the DADA buffer key from command line */
    key_t dada_key;
    if (sscanf(argv[1], "%x", &dada_key) != 1) {
        fprintf(stderr, "Error: Invalid DADA key.\n");
        return EXIT_FAILURE;
    }
    
    /* Create and connect the DADA Header/Data Unit (HDU) for reading */
    dada_hdu_t *hdu = dada_hdu_create(NULL);
    dada_hdu_set_key(hdu, dada_key);
    if (dada_hdu_connect(hdu) < 0) {
        fprintf(stderr, "Error: Could not connect to DADA buffer.\n");
        return EXIT_FAILURE;
    }
    if (dada_hdu_lock_read(hdu) < 0) {
        fprintf(stderr, "Error: Could not lock DADA buffer for reading.\n");
        return EXIT_FAILURE;
    }
    
    /* Get the size of a data block and compute the number of time samples.
     * Here we assume that each time sample consists of NANTS bytes (one per antenna).
     */
    //uint64_t block_size = ipcbuf_get_bufsz(hdu->data_block);
    uint64_t block_size = ipcbuf_get_bufsz((ipcbuf_t *)hdu->data_block);
    
    printf("Block size: %" PRIu64 " bytes\n", block_size);
    uint64_t samples_per_block = block_size / NANTS;
    printf("Time samples per block: %" PRIu64 "\n", samples_per_block);
    
    /* Allocate the visibility accumulation array.
     * We compute a full correlation matrix (including auto-correlations).
     */
    complex_d visibilities[NANTS][NANTS];
    memset(visibilities, 0, sizeof(visibilities));
    
    uint64_t integration_count = 0;
    
    /* Get the next data block from the DADA buffer */
    uint64_t bytes;
    char *data_block = ipcbuf_get_next_read((ipcbuf_t *)hdu->data_block, &bytes);
    if (!data_block) {
        fprintf(stderr, "Error: Could not get the next data block.\n");
        return EXIT_FAILURE;
    }
    
    /* Loop over each time sample in the block */
    for (uint64_t t = 0; t < samples_per_block; t++) {
        /* Each time sample consists of NANTS consecutive bytes */
        uint8_t *sample_ptr = (uint8_t *)(data_block + t * NANTS);
        int antenna_samples[NANTS][2]; // [antenna][0] = real, [antenna][1] = imag
        
        /* Decode each antenna's complex voltage sample */
        for (int ant = 0; ant < NANTS; ant++) {
            decode_sample(sample_ptr[ant], &antenna_samples[ant][0], &antenna_samples[ant][1]);
        }
        
        /* Compute cross-correlations for every pair of antennas.
         * For antennas i and j, the visibility is:
         *    V(i,j) += sample(i) * conj(sample(j))
         * where:
         *    sample(i) = a_re + i a_im, and
         *    conj(sample(j)) = b_re - i b_im.
         * Thus, the product is:
         *    a_re * b_re + a_im * b_im  (real part)
         *    a_im * b_re - a_re * b_im  (imaginary part)
         */
        for (int i = 0; i < NANTS; i++) {
            for (int j = i; j < NANTS; j++) {
                int a_re = antenna_samples[i][0], a_im = antenna_samples[i][1];
                int b_re = antenna_samples[j][0], b_im = antenna_samples[j][1];
                double prod_re = a_re * b_re + a_im * b_im;
                double prod_im = a_im * b_re - a_re * b_im;
                
                visibilities[i][j].re += prod_re;
                visibilities[i][j].im += prod_im;
                /* For off-diagonal elements, fill in the symmetric (conjugate) value */
                if (i != j) {
                    visibilities[j][i].re += prod_re;
                    visibilities[j][i].im -= prod_im;
                }
            }
        }
        integration_count++;
        
        /* Once we have accumulated the desired number of time samples, output
         * the averaged visibilities and reset the accumulation.
         */
        if (integration_count == INTEGRATION_SAMPLES) {
            printf("\nAveraged visibilities over %d samples:\n", INTEGRATION_SAMPLES);
            for (int i = 0; i < NANTS; i++) {
                for (int j = 0; j < NANTS; j++) {
                    double avg_re = visibilities[i][j].re / INTEGRATION_SAMPLES;
                    double avg_im = visibilities[i][j].im / INTEGRATION_SAMPLES;
                    printf("V[%d][%d] = %f + i%f\n", i, j, avg_re, avg_im);
                    /* Reset for the next integration period */
                    visibilities[i][j].re = 0.0;
                    visibilities[i][j].im = 0.0;
                }
            }
            integration_count = 0;
            /* Optionally, here you might write the averaged visibilities to disk or send them over a network */
        }
    }
    
    /* Mark the block as processed */
    ipcbuf_mark_cleared((ipcbuf_t *)hdu->data_block);
    
    
    /* Clean up the DADA resources */
    dada_hdu_unlock_read(hdu);
    dada_hdu_disconnect(hdu);
    dada_hdu_destroy(hdu);
    
    return EXIT_SUCCESS;
}
