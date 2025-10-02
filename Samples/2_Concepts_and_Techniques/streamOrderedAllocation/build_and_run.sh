#!/bin/bash

SEP="--------------------------------------------------------------------"

BUILD_ROOT=build
TARGET=streamOrderedAllocation

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

    rm -rf $BUILD_ROOT && mkdir -p $BUILD_ROOT && cd $BUILD_ROOT || exit

    cmake -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100" ..
    make -j8 || exit

    cd ..
else
    echo "$SEP"
    echo "Skipping build process"
    echo "$SEP"
fi

# run

# CMD="./$BUILD_ROOT/$TARGET" # only test the correctness of one kernel, default kernel7
CMD="./$BUILD_ROOT/$TARGET --shmoo" # benchmark all kernels with the array size goes

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