/* will reorder raw data for input to xgpu */
#define __USE_GNU
#define _GNU_SOURCE
#include <sched.h>
#include <time.h>
#include <sys/socket.h>
#include <math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <syslog.h>

#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "casm_capture.h"
#include "casm_def.h"

/* global variables */
int DEBUG = 0;
int STATS = 0;
const int nth = 4;

// Simplified antenna ordering array for 12 antennas
int * ant_order;
ant_order = (int *)malloc(sizeof(int)*12);
// Example ordering - modify these numbers as needed for your case
ant_order[0] = 3;   // first antenna in input goes to position 3 in output
ant_order[1] = 7;   // second antenna in input goes to position 7 in output
ant_order[2] = 0;   // etc...
ant_order[3] = 1;
ant_order[4] = 2;
ant_order[5] = 4;
ant_order[6] = 5;
ant_order[7] = 6;
ant_order[8] = 8;
ant_order[9] = 9;
ant_order[10] = 10;
ant_order[11] = 11;

// Modify the data struct to be simpler
struct data {
    char * in;
    char * out;
    int * ant_order;
    int n_threads;
    int thread_id;
};

void * massage (void *args) {
    struct data *d = args;
    int thread_id = d->thread_id;

    // set affinity
    const pthread_t pid = pthread_self();
    const int core_id = cores[thread_id];
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    const int set_result = pthread_setaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
    if (set_result != 0)
        syslog(LOG_ERR,"thread %d: setaffinity_np fail",thread_id);
    const int get_affinity = pthread_getaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
    if (get_affinity != 0) 
        syslog(LOG_ERR,"thread %d: getaffinity_np fail",thread_id);
    if (CPU_ISSET(core_id, &cpuset))
        if (DEBUG) syslog(LOG_DEBUG,"thread %d: successfully set thread",thread_id);

    char *in = (char *)d->in;
    char *out = (char *)d->out;
    int n_threads = d->n_threads;
    int * ao = d->ant_order;

    uint64_t oidx, iidx, ncpy = NCHAN;

    // Assuming 2048 time samples per block
    for (int i=thread_id*(2048/n_threads); i<(thread_id+1)*(2048/n_threads); i++) {
        for (int j=0; j<12; j++) {  // Loop over 12 antennas
            iidx = i*NANTS*UDP_DATA + j*NCHAN;  // Original position in input
            oidx = i*NANTS*UDP_DATA + ao[j]*NCHAN;  // New position in output
            memcpy(out + oidx, in + iidx, ncpy);
        }
    }

    int thread_result = 0;
    pthread_exit((void *) &thread_result);
}

void dsaX_dbgpu_cleanup (dada_hdu_t * in, int write);
int dada_bind_thread_to_core (int core);

void dsaX_dbgpu_cleanup (dada_hdu_t * in, int write)
{

    if (write==0) {
    
        if (dada_hdu_unlock_read (in) < 0)
            {
                syslog(LOG_ERR, "could not unlock read on hdu_in");
            }
        dada_hdu_destroy (in);

    }

    if (write==1) {

        if (dada_hdu_unlock_write (in) < 0)
            {
                syslog(LOG_ERR, "could not unlock write on hdu_in");
            }
        dada_hdu_destroy (in);

    }
}

void usage()
{
    fprintf (stdout,
             "dsaX_split [options]\n"
             " -c core   bind process to CPU core [no default]\n"
             " -d send debug messages to syslog\n"
             " -m multithread write\n"
             " -i in_key\n"
             " -o out_key\n"
             " -j in_key2\n"
             " -h print usage\n");
}

// MAIN

int main (int argc, char *argv[]) {
    
    // startup syslog message
    // using LOG_LOCAL0
    openlog ("dsaX_merge", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
    syslog (LOG_NOTICE, "Program started by User %d", getuid ());
    
    /* DADA Header plus Data Unit */
    dada_hdu_t* hdu_in = 0;
    dada_hdu_t* hdu_out = 0;
    dada_hdu_t* hdu_in2 = 0;

    // data block HDU keys
    key_t in_key = CAPTURE_BLOCK_KEY;
    key_t out_key = CAPTURED_BLOCK_KEY;
    key_t in_key2 = REORDER_BLOCK_KEY2;
    
    // command line arguments
    int core = -1;
    int arg = 0;
    int mwrite = 0;
    
    while ((arg=getopt(argc,argv,"c:i:o:j:dmh")) != -1)
        {
            switch (arg)
            {
                case 'c':
                    if (optarg)
                    {
                        core = atoi(optarg);
                        break;
                    }
                    else
                    {
                        syslog(LOG_ERR,"-c flag requires argument");
                        usage();
                        return EXIT_FAILURE;
                    }
                case 'i':
                    if (optarg)
                    {
                        if (sscanf (optarg, "%x", &in_key) != 1) {
                            syslog(LOG_ERR, "could not parse key from %s\n", optarg);
                            return EXIT_FAILURE;
                        }
                        break;
                    }
                    else
                    {
                        syslog(LOG_ERR,"-i flag requires argument");
                        usage();
                        return EXIT_FAILURE;
                    }
                case 'o':
                    if (optarg)
                    {
                        if (sscanf (optarg, "%x", &out_key) != 1) {
                            syslog(LOG_ERR, "could not parse key from %s\n", optarg);
                            return EXIT_FAILURE;
                        }
                        break;
                    }
                    else
                    {
                        syslog(LOG_ERR,"-o flag requires argument");
                        usage();
                        return EXIT_FAILURE;
                    }
                case 'j':
                    if (optarg)
                    {
                        if (sscanf (optarg, "%x", &in_key2) != 1) {
                            syslog(LOG_ERR, "could not parse key from %s\n", optarg);
                            return EXIT_FAILURE;
                        }
                        break;
                    }
                    else
                    {
                        syslog(LOG_ERR,"-j flag requires argument");
                        usage();
                        return EXIT_FAILURE;
                    }
                case 'd':
                    DEBUG=1;
                    syslog (LOG_DEBUG, "Will excrete all debug messages");
                    break;
                case 'm':
                    mwrite=1;
                    syslog (LOG_INFO, "Will do multithread write");
                    break;
                case 'h':
                    usage();
                    return EXIT_SUCCESS;
            }
        }

    // Bind to cpu core
    if (core >= 0)
    {
        if (dada_bind_thread_to_core(core) < 0)
            syslog(LOG_ERR,"failed to bind to core %d", core);
        syslog(LOG_NOTICE,"bound to core %d", core);
    }

    
    // DADA stuff
    
    syslog (LOG_INFO, "creating in and out hdus");
    
    multilog_t* log = multilog_open("casm_merge", 0);
    hdu_in = dada_hdu_create(log);
    dada_hdu_set_key (hdu_in, in_key);
    if (dada_hdu_connect (hdu_in) < 0) {
        syslog (LOG_ERR,"could not connect to dada buffer in");
        return EXIT_FAILURE;
    }
    if (dada_hdu_lock_read (hdu_in) < 0) {
        syslog (LOG_ERR,"could not lock to dada buffer in");
        return EXIT_FAILURE;
    }

    hdu_out  = dada_hdu_create (log);
    dada_hdu_set_key (hdu_out, out_key);
    if (dada_hdu_connect (hdu_out) < 0) {
        syslog (LOG_ERR,"could not connect to output  buffer");
        return EXIT_FAILURE;
    }
    if (dada_hdu_lock_write(hdu_out) < 0) {
        syslog (LOG_ERR, "could not lock to output buffer");
        return EXIT_FAILURE;
    }

    hdu_in2  = dada_hdu_create (log);
    dada_hdu_set_key (hdu_in2, in_key2);
    if (dada_hdu_connect (hdu_in2) < 0) {
        syslog (LOG_ERR,"could not connect to input  buffer2");
        return EXIT_FAILURE;
    }
    if (dada_hdu_lock_read(hdu_in2) < 0) {
        syslog (LOG_ERR, "could not lock to input buffer2");
        return EXIT_FAILURE;
    }
    
    uint64_t header_size = 0;

    // deal with headers
    char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
    if (!header_in)
    {
        syslog(LOG_ERR, "could not read next header");
        dsaX_dbgpu_cleanup (hdu_in,0);
        dsaX_dbgpu_cleanup (hdu_in2,0);
        dsaX_dbgpu_cleanup (hdu_out,1);
        
        return EXIT_FAILURE;
    }
    if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
        syslog (LOG_ERR, "could not mark header block cleared");
        dsaX_dbgpu_cleanup (hdu_in,0);
        dsaX_dbgpu_cleanup (hdu_in2,0);
        dsaX_dbgpu_cleanup (hdu_out,1);
        return EXIT_FAILURE;
    }
    header_in = ipcbuf_get_next_read (hdu_in2->header_block, &header_size);
    if (!header_in)
    {
        syslog(LOG_ERR, "could not read next header");
        dsaX_dbgpu_cleanup (hdu_in,0);
        dsaX_dbgpu_cleanup (hdu_in2,0);
        dsaX_dbgpu_cleanup (hdu_out,1);
        
        return EXIT_FAILURE;
    }
    if (ipcbuf_mark_cleared (hdu_in2->header_block) < 0)
    {
        syslog (LOG_ERR, "could not mark header block cleared");
        dsaX_dbgpu_cleanup (hdu_in,0);
        dsaX_dbgpu_cleanup (hdu_in2,0);
        dsaX_dbgpu_cleanup (hdu_out,1);
        return EXIT_FAILURE;
    }

    char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
    if (!header_out)
    {
        syslog(LOG_ERR, "could not get next header block [output]");
        dsaX_dbgpu_cleanup (hdu_in,0);
        dsaX_dbgpu_cleanup (hdu_in2,0);
        dsaX_dbgpu_cleanup (hdu_out,1);
        return EXIT_FAILURE;
    }
    memcpy (header_out, header_in, header_size);
    if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
        syslog (LOG_ERR, "could not mark header block filled [output]");
        dsaX_dbgpu_cleanup (hdu_in,0);
        dsaX_dbgpu_cleanup (hdu_in2,0);
        dsaX_dbgpu_cleanup (hdu_out,1);
        return EXIT_FAILURE;
    }
    
    // record STATE info
    syslog(LOG_INFO,"dealt with dada stuff - now in LISTEN state");

    // get block sizes and allocate memory
    uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
    uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
    syslog(LOG_INFO, "main: have input and output block sizes %llu %llu\n",block_size,block_out);
    uint64_t  bytes_read = 0;
    char * block1, * block2, * o1, * o2;
    char * output = (char *)malloc(sizeof(char)*block_out);
    uint64_t written, block_id;

    // set up threads
    struct data args[8];
    pthread_t threads[8];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    void* result=0;
    
    // send through fake blocks

    /*  if (fake>0) {
        syslog(LOG_INFO,"sending %d fake blocks",fake);
        for (int i=0;i<fake;i++) {
            o1 = ipcio_open_block_write (hdu_out->data_block, &block_id);
            memcpy(o1, output, block_out);
            ipcio_close_block_write (hdu_out->data_block, block_out);
            usleep(10000);
        }
        syslog(LOG_INFO,"Finished with fake blocks");
    }*/
    
    
    
    // set up

    int observation_complete=0;
    int blocks = 0;
    int started = 0;


    
    syslog(LOG_INFO, "starting observation");

    while (!observation_complete) {

        // open block
        
        block1 = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
        block2 = ipcio_open_block_read (hdu_in2->data_block, &bytes_read, &block_id);

        if (started==0) {
            syslog(LOG_INFO,"now in RUN state");
            started=1;
        }

        
        // DO STUFF

        // copy to output buffer
        
        if (mwrite) {
            o1 = ipcio_open_block_write (hdu_out->data_block, &block_id);
        }
        
        // set up data structure
        for (int i=0; i<nth; i++) {
            args[i].in = block1;
            args[i].in2 = block2;
            args[i].ant_order = ant_order;
            
            if (mwrite) 
                args[i].out = o1;	
            else
                args[i].out = output;

            args[i].n_threads = nth;
            args[i].thread_id = i;
        }
        
        //syslog(LOG_INFO, "creating threads");
        
        for(int i=0; i<nth; i++){
            if (pthread_create(&threads[i], &attr, &massage, (void *)(&args[i]))) {
                syslog(LOG_ERR,"Failed to create massage thread %d\n", i);
            }
        }
        
        pthread_attr_destroy(&attr);
        if (DEBUG) syslog(LOG_DEBUG,"threads kinda running");
        
        for(int i=0; i<nth; i++){
            pthread_join(threads[i], &result);
            if (DEBUG) syslog(LOG_DEBUG,"joined thread %d",i);
        }
        
        
        if (!mwrite) {
            written = ipcio_write (hdu_out->data_block, output, block_out);
        }
        else {
            ipcio_close_block_write (hdu_out->data_block, block_out);
        }

        if (blocks % 10 == 0)
            syslog(LOG_INFO, "written block %d",blocks);      
        blocks++;
        
        
        if (bytes_read < block_size)
            observation_complete = 1;            
        
        ipcio_close_block_read (hdu_in->data_block, bytes_read);
        ipcio_close_block_read (hdu_in2->data_block, bytes_read);

    }

    free(output);
    dsaX_dbgpu_cleanup (hdu_in,0);
    dsaX_dbgpu_cleanup (hdu_in2,0);
    dsaX_dbgpu_cleanup (hdu_out,1);
    
}


