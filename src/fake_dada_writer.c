#include <stdio.h>
#include <string.h>
#include "dada_hdu.h"
#include "ipcio.h"

int main() {
  dada_hdu_t* hdu = dada_hdu_create(0);
  dada_hdu_set_key(hdu, 0x1234);
  dada_hdu_connect(hdu);
  dada_hdu_lock_write(hdu);

  uint64_t block_size = ipcbuf_get_bufsz(hdu->data_block);
  char* block = ipcio_open_block_write(hdu->data_block, NULL);
  memset(block, 0, block_size);  // Fill with zeros or synthetic data
  ipcio_close_block_write(hdu->data_block, block_size);

  dada_hdu_unlock_write(hdu);
  dada_hdu_disconnect(hdu);
  return 0;
}