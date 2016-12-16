// Copyright 2016 Yahoo Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
// Copyright (C) 1998-2003 Fast Search & Transfer ASA
// Copyright (C) 2003 Overture Services Norway AS

#include <vespa/log/log.h>
#include <vespa/searchcommon/common/undefinedvalues.h>
#include <vespa/searchsummary/docsummary/resultpacker.h>

LOG_SETUP(".searchlib.docsummary.resultpacker");

namespace search {
namespace docsummary {

void
ResultPacker::WarnType(ResType type) const
{
    LOG(debug,
        "ResultPacker: got '%s', expected '%s' "
        "(fields are binary compatible)",
        GetResTypeName(type),
        GetResTypeName(_cfgEntry->_type));
}

bool ResultPacker::CheckEntry(ResType type)
{
    if (_error)
        return false;

    bool rc = (_cfgEntry != NULL &&
               IsBinaryCompatible(_cfgEntry->_type, type));

    if (rc) {
        if (_cfgEntry->_type != type) {
            WarnType(type);
        }
        _cfgEntry = _resClass->GetEntry(++_entryIdx);
    } else {
        SetFormatError(type);
    }

    return rc;
}

void
ResultPacker::SetFormatError(ResType type)
{
    _error = true;

    if (_cfgEntry != NULL) {
        LOG(error,
            "ResultPacker: format error: got '%s', expected '%s'",
            GetResTypeName(type),
            GetResTypeName(_cfgEntry->_type));
    } else {
        LOG(error,
            "ResultPacker: format error: "
            "got '%s', no more fields expected", GetResTypeName(type));
    }
}


ResultPacker::ResultPacker(const ResultConfig *resConfig)
    : _buf(32768),
      _cbuf(32768),
      _resConfig(resConfig),
      _resClass(NULL),
      _entryIdx(0),
      _cfgEntry(NULL),
      _error(true)
{
}


ResultPacker::~ResultPacker()
{
}

void
ResultPacker::InitPlain()
{
    _buf.reset();
}

bool
ResultPacker::Init(uint32_t classID)
{
    _buf.reset();
    _resClass = (_resConfig != NULL) ?
                _resConfig->LookupResultClass(classID) : NULL;
    _entryIdx = 0;
    if (_resClass != NULL) {
        uint32_t id = _resClass->GetClassID();
        _buf.append(&id, sizeof(id));
        _cfgEntry = _resClass->GetEntry(_entryIdx);
        _error = false;
    } else {
        _cfgEntry = NULL;
        _error = true;

        LOG(error, "ResultPacker: resultclass %d does not exist", classID);
    }

    return !_error;
}


bool
ResultPacker::AddEmpty()
{
    if (!_error && _cfgEntry != NULL) {
        switch (_cfgEntry->_type) {
        case RES_INT:         return AddInteger(search::attribute::getUndefined<int32_t>());
        case RES_SHORT:       return AddShort(search::attribute::getUndefined<int16_t>());
        case RES_BYTE:        return AddByte(search::attribute::getUndefined<int8_t>());
        case RES_FLOAT:       return AddFloat(search::attribute::getUndefined<float>());
        case RES_DOUBLE:      return AddDouble(search::attribute::getUndefined<double>());
        case RES_INT64:       return AddInt64(search::attribute::getUndefined<int64_t>());
        case RES_STRING:      return AddString(NULL, 0);
        case RES_DATA:        return AddData(NULL, 0);
        case RES_XMLSTRING:
        case RES_JSONSTRING:
        case RES_FEATUREDATA:
        case RES_LONG_STRING: return AddLongString(NULL, 0);
        case RES_LONG_DATA:   return AddLongData(NULL, 0);
        }
    }
    return AddInteger(0); // to provoke error condition
}


bool
ResultPacker::AddByte(uint8_t value)
{
    if (CheckEntry(RES_BYTE))
        AddByteForce(value);
    return !_error;
}

void
ResultPacker::AddByteForce(uint8_t value)
{
    _buf.append(&value, sizeof(value));
}

bool
ResultPacker::AddShort(uint16_t value)
{
    if (CheckEntry(RES_SHORT))
        AddShortForce(value);
    return !_error;
}

void
ResultPacker::AddShortForce(uint16_t value)
{
    _buf.append(&value, sizeof(value));
}


bool
ResultPacker::AddInteger(uint32_t value)
{
    if (CheckEntry(RES_INT))
        AddIntegerForce(value);
    return !_error;
}

void
ResultPacker::AddIntegerForce(uint32_t value)
{
    _buf.append(&value, sizeof(value));
}


bool
ResultPacker::AddFloat(float value)
{
    if (CheckEntry(RES_FLOAT))
        _buf.append(&value, sizeof(value));
    return !_error;
}


bool
ResultPacker::AddDouble(double value)
{
    if (CheckEntry(RES_DOUBLE))
        _buf.append(&value, sizeof(value));
    return !_error;
}


bool
ResultPacker::AddInt64(uint64_t value)
{
    if (CheckEntry(RES_INT64))
        _buf.append(&value, sizeof(value));
    return !_error;
}


bool
ResultPacker::AddString(const char *str, uint32_t slen)
{
    if (CheckEntry(RES_STRING))
        AddStringForce(str, slen);
    return !_error;
}

void
ResultPacker::AddStringForce(const char *str, uint32_t slen)
{
    uint16_t len = slen;
    _buf.append(&len, sizeof(len));
    _buf.append(str, len);
}


bool
ResultPacker::AddData(const char *buf, uint32_t buflen)
{
    if (CheckEntry(RES_DATA)) {
        uint16_t len = buflen;
        _buf.append(&len, sizeof(len));
        _buf.append(buf, len);
    }
    return !_error;
}


bool
ResultPacker::AddLongString(const char *str, uint32_t slen)
{
    if (CheckEntry(RES_LONG_STRING)) {
        _buf.append(&slen, sizeof(slen));
        _buf.append(str, slen);
    }
    return !_error;
}


bool
ResultPacker::AddLongData(const char *buf, uint32_t buflen)
{
    if (CheckEntry(RES_LONG_DATA)) {
        _buf.append(&buflen, sizeof(buflen));
        _buf.append(buf, buflen);
    }
    return !_error;
}


bool
ResultPacker::GetDocsumBlob(const char **buf, uint32_t *buflen)
{
    if (!_error &&
        _entryIdx != _resClass->GetNumEntries())
    {
        _error = true;
        LOG(error,
            "ResultPacker: format error: %d fields are missing",
            _resClass->GetNumEntries() - _entryIdx);
    }
    if (_error) {
        *buf    = NULL;
        *buflen = 0;
        return false;
    } else {
        *buf    = _buf.GetDrainPos();
        *buflen = _buf.GetUsedLen();
        return true;
    }
}

void
ResultPacker::GetDocsumBlobForce(const char **buf, uint32_t *buflen)
{
    *buf    = _buf.GetDrainPos();
    *buflen = _buf.GetUsedLen();
}

}
}
