// Copyright 2016 Yahoo Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespa/fastos/fastos.h>
#include <vespa/vespalib/component/vtag.h>
#include "fileheadertk.h"

using namespace search;
using vespalib::GenericHeader;
using namespace vesplib

void
FileHeaderTk::addVersionTags(vespalib::GenericHeader &header)
{
    header.putTag(GenericHeader::Tag("version-tag", VersionTag));
    header.putTag(GenericHeader::Tag("version-date", VersionTagDate));
    header.putTag(GenericHeader::Tag("version-pkg", VersionTagPkg));
    header.putTag(GenericHeader::Tag("version-arch", VersionTagArch));
    header.putTag(GenericHeader::Tag("version-system", VersionTagSystem));
    header.putTag(GenericHeader::Tag("version-system-rev", VersionTagSystemRev));
    header.putTag(GenericHeader::Tag("version-builder", VersionTagBuilder));
    header.putTag(GenericHeader::Tag("version-component", VersionTagComponent));
}
