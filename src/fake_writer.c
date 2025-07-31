#include <stdio.h>
#include <string.h>
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

  fprintf(stderr, "Getting block size...\n");
  ipcbuf_t* buf = &hdu->data_block->buf;
  if (!buf) {
    fprintf(stderr, "data_block->buf is NULL\n");
    return 1;
  }

  uint64_t block_size = ipcbuf_get_bufsz(buf);
  fprintf(stderr, "Block size: %lu bytes\n", block_size);

  fprintf(stderr, "Opening block for write...\n");
  uint64_t block_id;
  char* block = ipcio_open_block_write(hdu->data_block, &block_id);
  fprintf(stderr, "Opened block ID: %lu\n", block_id);

if (!block) {
  fprintf(stderr, "ipcio_open_block_write returned NULL\n");
  return 1;
}

  fprintf(stderr, "Zeroing buffer and closing write block...\n");
  memset(block, 0, block_size);
  ipcio_close_block_write(hdu->data_block, block_size);

uint64_t block_size = ipcbuf_get_bufsz(&hdu->data_block->buf);
fprintf(stderr, "Block size: %lu bytes\n", block_size);

// ADD THESE THREE LINES TO WRITE A DUMMY HEADER
fprintf(stderr, "Writing dummy header...\n");
char* header = ipcbuf_get_next_write(hdu->header_block);
strcpy(header, "HDR_SIZE 4096\n"); // A minimal, valid header
ipcbuf_mark_filled(hdu->header_block, strlen(header));
  
// ADD THIS LINE
while (1) { 
    fprintf(stderr, "Opening block for write...\n");
    uint64_t block_id;
    char* block = ipcio_open_block_write(hdu->data_block, &block_id);
    if (!block) {
        fprintf(stderr, "ipcio_open_block_write returned NULL\n");
        // Maybe add a sleep here to prevent a tight error loop
        sleep(1); 
        continue; // TRY AGAIN
    }
    fprintf(stderr, "Opened block ID: %lu\n", block_id);


    fprintf(stderr, "Zeroing buffer and closing write block...\n");
    memset(block, 0, block_size);
    ipcio_close_block_write(hdu->data_block, block_size);

    fprintf(stderr, "Wrote block %lu. Looping...\n", block_id); 
// ADD THIS CLOSING BRACE
} 
  
  fprintf(stderr, "Unlocking and disconnecting...\n");
  dada_hdu_unlock_write(hdu);
  dada_hdu_disconnect(hdu);

  fprintf(stderr, "Done.\n");
  return 0;
}
