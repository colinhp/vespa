// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <cstdint>

/**
 * This class is used internally by the @ref FNET_Transport to keep
 * track of the current configuration.
 **/
class FNET_Config
{
public:
    uint32_t  _iocTimeOut;
    uint32_t  _maxInputBufferSize;
    uint32_t  _maxOutputBufferSize;
    bool      _tcpNoDelay;

    FNET_Config();
};
