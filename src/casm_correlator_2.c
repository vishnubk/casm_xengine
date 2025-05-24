/*
 * dsaX_corr_freq_save_bin.c
 *
 * Example correlator that reads voltage data from a DADA buffer,
 * cross correlates (computes visibilities) for 12 antennas (single polarization)
 * across multiple frequency channels, and then averages those visibilities over time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <unistd.h>  // Include for getopt and sleep
#include "dada_hdu.h"
#include "ipcbuf.h"

#define NANTS 12                   // Number of antennas
#define NCHAN 3072                 // Number of frequency channels
#define INTEGRATION_SAMPLES 1024   // Number of time samples to average over

/* Structure for accumulating complex visibilities */
typedef struct {
    double re;
    double im;
} complex_d;

/*
 * decode_sample: Decode an 8-bit sample (4+4-bit) into its real and imaginary parts.
 * The 4-bit values are in two's complement.
 */
static inline void decode_sample(uint8_t sample, int *re, int *im) {
    int r = (sample >> 4) & 0x0F;
    int i = sample & 0x0F;
    if (r & 0x08) r -= 16;
    if (i & 0x08) i -= 16;
    *re = r;
    *im = i;
}

int main(int argc, char **argv) {
    key_t dada_key = 0;
    int opt;

    // Parse command-line options
    while ((opt = getopt(argc, argv, "k:")) != -1) {
        switch (opt) {
            case 'k':
                if (sscanf(optarg, "%x", &dada_key) != 1) {
                    fprintf(stderr, "Error: Invalid DADA key.\n");
                    return EXIT_FAILURE;
                }
                break;
            default:
                fprintf(stderr, "Usage: %s -k <DADA key in hex>\n", argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (dada_key == 0) {
        fprintf(stderr, "Usage: %s -k <DADA key in hex>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Create and connect to the DADA buffer
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

    uint64_t block_size = ipcbuf_get_bufsz((ipcbuf_t *)hdu->data_block);
    uint64_t bytes_per_time = NCHAN * NANTS;
    printf("Block size: %" PRIu64 " bytes\n", block_size);

    // Allocate an accumulation array for visibilities
    complex_d visibilities[NCHAN][NANTS][NANTS];
    memset(visibilities, 0, sizeof(visibilities));
    uint64_t integration_count = 0;

    // Open the output binary file in append mode
    FILE *outfile = fopen("visibilities.bin", "ab");
    if (!outfile) {
        fprintf(stderr, "Error: Could not open output binary file.\n");
        return EXIT_FAILURE;
    }

    // Continuous processing loop
    while (1) {
        char *data_block;
        uint64_t bytes_read;

        // Wait for the next data block
        while ((data_block = ipcbuf_get_next_read((ipcbuf_t *)hdu->data_block, &bytes_read)) == NULL) {
            printf("Waiting for new data block...\n");
            sleep(1);
        }

        printf("Retrieved a new data block of %" PRIu64 " bytes\n", bytes_read);
        uint64_t time_samples = bytes_read / bytes_per_time;
        printf("Processing %lu time samples in this block\n", time_samples);

        // Process the current block
        for (uint64_t t = 0; t < time_samples; t++) {
            for (int ch = 0; ch < NCHAN; ch++) {
                uint8_t *sample_ptr = (uint8_t *)(data_block + t * bytes_per_time + ch * NANTS);
                int voltages[NANTS][2];

                for (int ant = 0; ant < NANTS; ant++) {
                    decode_sample(sample_ptr[ant], &voltages[ant][0], &voltages[ant][1]);
                }

                for (int i = 0; i < NANTS; i++) {
                    for (int j = i; j < NANTS; j++) {
                        double prod_re = voltages[i][0] * voltages[j][0] + voltages[i][1] * voltages[j][1];
                        double prod_im = voltages[i][1] * voltages[j][0] - voltages[i][0] * voltages[j][1];
                        visibilities[ch][i][j].re += prod_re;
                        visibilities[ch][i][j].im += prod_im;
                        if (i != j) {
                            visibilities[ch][j][i].re += prod_re;
                            visibilities[ch][j][i].im -= prod_im;
                        }
                    }
                }
            }

            integration_count++;

            if (integration_count == INTEGRATION_SAMPLES) {
                printf("Completed integration of %d samples. Writing to file.\n", INTEGRATION_SAMPLES);
                for (int ch = 0; ch < NCHAN; ch++) {
                    for (int i = 0; i < NANTS; i++) {
                        for (int j = 0; j < NANTS; j++) {
                            double avg_re = visibilities[ch][i][j].re / INTEGRATION_SAMPLES;
                            double avg_im = visibilities[ch][i][j].im / INTEGRATION_SAMPLES;
                            fwrite(&avg_re, sizeof(double), 1, outfile);
                            fwrite(&avg_im, sizeof(double), 1, outfile);
                            visibilities[ch][i][j].re = 0.0;
                            visibilities[ch][i][j].im = 0.0;
                        }
                    }
                }
                fflush(outfile);
                integration_count = 0;
            }
        }

        // Mark block as cleared so new data can be written
        ipcbuf_mark_cleared((ipcbuf_t *)hdu->data_block);
    }

    // Cleanup (will never be reached unless interrupted)
    dada_hdu_unlock_read(hdu);
    dada_hdu_disconnect(hdu);
    dada_hdu_destroy(hdu);
    fclose(outfile);

    return EXIT_SUCCESS;
}
