#!/bin/bash

SEP="--------------------------------------------------------------------"

BUILD_ROOT=build
TARGET=matrixMul_nvrtc

# default not skip any step
SKIP_BUILD=false
SKIP_RUN=false
SKIP_PROFILE=true

# parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            ;;
        --skip-run)
            SKIP_RUN=true
            ;;
        --skip-profile)
            SKIP_PROFILE=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done


# build
if [ "$SKIP_BUILD" = false ]; then
    echo "$SEP"
    echo "Building ${TARGET}"
    echo "$SEP"

    mkdir -p $BUILD_ROOT && cd $BUILD_ROOT || exit

    cmake ..
    make -j8 || exit

    cd ..
else
    echo "$SEP"
    echo "Skipping build process"
    echo "$SEP"
fi

# run

cd $BUILD_ROOT

CMD="./$TARGET"

if [ "$SKIP_RUN" = false ]; then
    echo "$SEP"
    echo "Running ${TARGET}"
    echo "$SEP"
    $CMD
else
    echo "$SEP"
    echo "Skipping run process"
    echo "$SEP"
fi

# profile
if [ "$SKIP_PROFILE" = false ]; then
    echo "$SEP"
    echo "Profiling ${TARGET}"
    echo "$SEP"

    nsys profile \
        --force-overwrite true \
        -o ${TARGET}.nsys-rep \
        --capture-range=cudaProfilerApi \
        $CMD
else
    echo "$SEP"
    echo "Skipping profiling process"
    echo "$SEP"
fi

echo "$SEP"
echo "Done"
echo "$SEP"