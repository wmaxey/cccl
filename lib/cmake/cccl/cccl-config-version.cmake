set(CCCL_VERSION_MAJOR 2)
set(CCCL_VERSION_MINOR 9)
set(CCCL_VERSION_PATCH 5)
set(CCCL_VERSION_TWEAK 0)

set(CCCL_VERSION "${CCCL_VERSION_MAJOR}.${CCCL_VERSION_MINOR}.${CCCL_VERSION_PATCH}.${CCCL_VERSION_TWEAK}")

set(PACKAGE_VERSION ${CCCL_VERSION})
set(PACKAGE_VERSION_COMPATIBLE FALSE)
set(PACKAGE_VERSION_EXACT FALSE)
set(PACKAGE_VERSION_UNSUITABLE FALSE)

# Semantic versioning:
if(PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION)
  if(CCCL_VERSION_MAJOR VERSION_EQUAL PACKAGE_FIND_VERSION_MAJOR)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  endif()

  if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
