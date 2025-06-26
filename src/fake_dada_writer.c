#include "dada_hdu.h"
#include "ipcio.h"
#include <string.h>

int main() {
  dada_hdu_t* hdu = dada_hdu_create(0);
  dada_hdu_set_key(hdu, 0x1234);
  dada_hdu_connect(hdu);
  dada_hdu_lock_write(hdu);

  char* block = ipcio_open_block_write(hdu->data_block, NULL);
  uint64_t block_size = ipcbuf_get_bufsz(hdu->data_block);
  memset(block, 0, block_size);
  ipcio_close_block_write(hdu->data_block, BLOCK_SIZE);

  dada_hdu_unlock_write(hdu);
  dada_hdu_disconnect(hdu);
  return 0;
}
