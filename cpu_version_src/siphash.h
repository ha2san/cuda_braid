#pragma once

#include <stddef.h>
#include <assert.h>
#include <stdint.h>

void siphash(const void *in, const size_t inlen, const void *k, uint8_t *out,
            const size_t outlen);
