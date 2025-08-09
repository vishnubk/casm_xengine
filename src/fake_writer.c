#include <stdio.h>
#include <string.h>
#include <unistd.h> // Required for sleep()
#include <getopt.h>
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"

void usage() {
    fprintf(stderr, "Usage: fake_writer [options]\n");
    fprintf(stderr, " -v, --verbose    Enable verbose output\n");
    fprintf(stderr, " -h, --help       Show this help message\n");
}

int main(int argc, char *argv[]) {
  int verbose = 0;
  int opt;
  
  // Parse command line arguments
  while ((opt = getopt(argc, argv, "vh")) != -1) {
    switch (opt) {
      case 'v':
        verbose = 1;
        break;
      case 'h':
        usage();
        return 0;
      default:
        usage();
        return 1;
    }
  }

  if (verbose) fprintf(stderr, "Creating HDU...\n");
  dada_hdu_t* hdu = dada_hdu_create(0);
  if (!hdu) {
    fprintf(stderr, "Failed to create DADA HDU\n");
    return 1;
  }

  if (verbose) fprintf(stderr, "Setting key...\n");
  dada_hdu_set_key(hdu, 0xdaaa);

  if (verbose) fprintf(stderr, "Connecting HDU...\n");
  if (dada_hdu_connect(hdu) < 0) {
    fprintf(stderr, "Failed to connect to HDU\n");
    return 1;
  }

  if (verbose) fprintf(stderr, "Locking HDU for write...\n");
  if (dada_hdu_lock_write(hdu) < 0) {
    fprintf(stderr, "Failed to lock HDU for writing\n");
    return 1;
  }

  // Get the block size once, before the loop starts.
  uint64_t block_size = ipcbuf_get_bufsz(&hdu->data_block->buf);
  if (verbose) fprintf(stderr, "Block size: %lu bytes\n", block_size);

  // Write the dummy header ONCE before writing any data.
  if (verbose) fprintf(stderr, "Writing dummy header...\n");
  char* header = ipcbuf_get_next_write(hdu->header_block);
  strcpy(header, "HDR_SIZE 4096\n"); // A minimal, valid header
  ipcbuf_mark_filled(hdu->header_block, strlen(header));

  // Loop forever, writing data blocks.
  uint64_t block_count = 0;
  while (1) {
    uint64_t block_id;
    char* block = ipcio_open_block_write(hdu->data_block, &block_id);

    if (!block) {
        if (verbose) fprintf(stderr, "ipcio_open_block_write returned NULL, waiting...\n");
        sleep(1);
        continue; // Try again
    }
    
    // This part is inside the loop
    if (verbose) fprintf(stderr, "Opened block ID: %lu\n", block_id);
    if (verbose) fprintf(stderr, "Zeroing buffer and closing write block...\n");
    memset(block, 0, block_size);
    ipcio_close_block_write(hdu->data_block, block_size);
    block_count++;
    
    // Only print every 10th block to reduce spam
    if (verbose || block_count % 10 == 0) {
        fprintf(stderr, "Wrote block %lu (total: %lu)\n", block_id, block_count);
    }
  }

  // The program will never reach this part in the loop version.
  if (verbose) fprintf(stderr, "Unlocking and disconnecting...\n");
  dada_hdu_unlock_write(hdu);
  dada_hdu_disconnect(hdu);
  if (verbose) fprintf(stderr, "Done.\n");

  return 0;
}