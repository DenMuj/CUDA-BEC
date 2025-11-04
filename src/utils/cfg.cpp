#include "cfg.h"

/**
 * @brief Configuration file parsing.
 *    cfg_file - stands for a configuration file, which is supplied on a command
 *    line
 */

int cfg_size;
char cfg_key[256][256];
char cfg_val[256][256];

/**
 * @brief Configuration file parsing.
 * @param cfg_file Configuration file
 * @return Configuration size
 */
int cfg_init(const char *cfg_file) {
    FILE *file;
    char buf[256];

    file = fopen(cfg_file, "r");
    if (!file)
        return 0;

    cfg_size = 0;
    while (fgets(buf, 256, file) != NULL) {
        if (sscanf(buf, "%s = %s", cfg_key[cfg_size], cfg_val[cfg_size]) == 2)
            cfg_size++;
    }

    fclose(file);

    return cfg_size;
}

/**
 * @brief Configuration property value.
 * @param key Property
 * @return Property value
 */
const char *cfg_read(const char *key) {
    int i;

    for (i = 0; i < cfg_size; i++)
        if (!strcmp(key, cfg_key[i]))
            return cfg_val[i];

    return NULL;
}