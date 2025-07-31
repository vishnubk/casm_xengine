#include <stdio.h>
#include <string.h>
#include <unistd.h> // Required for sleep()
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"

int main() {
  fprintf(stderr, "Creating HDU...\n");
  dada_hdu_t* hdu = dada_hdu_create(0);
  if (!hdu) {
    fprintf(stderr, "Failed to create DADA HDU\n");
    return 1;
  }

  fprintf(stderr, "Setting key...\n");
  dada_hdu_set_key(hdu, 0xdaaa);

  fprintf(stderr, "Connecting HDU...\n");
  if (dada_hdu_connect(hdu) < 0) {
    fprintf(stderr, "Failed to connect to HDU\n");
    return 1;
  }

  fprintf(stderr, "Locking HDU for write...\n");
  if (dada_hdu_lock_write(hdu) < 0) {
    fprintf(stderr, "Failed to lock HDU for writing\n");
    return 1;
  }

  // Get the block size once, before the loop starts.
  uint64_t block_size = ipcbuf_get_bufsz(&hdu->data_block->buf);
  fprintf(stderr, "Block size: %lu bytes\n", block_size);

  // Write the dummy header ONCE before writing any data.
  fprintf(stderr, "Writing dummy header...\n");
  char* header = ipcbuf_get_next_write(hdu->header_block);
  strcpy(header, "HDR_SIZE 4096\n"); // A minimal, valid header
  ipcbuf_mark_filled(hdu->header_block, strlen(header));

  // Loop forever, writing data blocks.
  while (1) {
    uint64_t block_id;
    char* block = ipcio_open_block_write(hdu->data_block, &block_id);

    if (!block) {
        fprintf(stderr, "ipcio_open_block_write returned NULL, waiting...\n");
        sleep(1);
        continue; // Try again
    }
    
    // This part is inside the loop
    fprintf(stderr, "Opened block ID: %lu\n", block_id);
    fprintf(stderr, "Zeroing buffer and closing write block...\n");
    memset(block, 0, block_size);
    ipcio_close_block_write(hdu->data_block, block_size);
    fprintf(stderr, "Wrote block %lu. Looping...\n", block_id);
  }

  // The program will never reach this part in the loop version.
  fprintf(stderr, "Unlocking and disconnecting...\n");
  dada_hdu_unlock_write(hdu);
  dada_hdu_disconnect(hdu);
  fprintf(stderr, "Done.\n");

  return 0;
}