#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int cfg_size;
extern char cfg_key[256][256];
extern char cfg_val[256][256];

int cfg_init(const char *);
const char *cfg_read(const char *);