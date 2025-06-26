#include <stdio.h>
#include <string.h>
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"

int main() {
  dada_hdu_t* hdu = dada_hdu_create(0);
  if (!hdu) {
    fprintf(stderr, "Failed to create DADA HDU\n");
    return 1;
  }

  dada_hdu_set_key(hdu, 0x1234);
  if (dada_hdu_connect(hdu) < 0) {
    fprintf(stderr, "Failed to connect to HDU\n");
    return 1;
  }

  if (dada_hdu_lock_write(hdu) < 0) {
    fprintf(stderr, "Failed to lock HDU for writing\n");
    return 1;
  }

  uint64_t block_size = ipcbuf_get_bufsz(hdu->data_block->buf);
  char* block = ipcio_open_block_write(hdu->data_block, NULL);
  if (!block) {
    fprintf(stderr, "Failed to open write block\n");
    return 1;
  }

  memset(block, 0, block_size);
  ipcio_close_block_write(hdu->data_block, block_size);

  dada_hdu_unlock_write(hdu);
  dada_hdu_disconnect(hdu);
  return 0;
}
