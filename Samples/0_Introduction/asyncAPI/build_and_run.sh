
SEP="--------------------------------------------------------------------"
TARGET=asyncAPI

echo "$SEP"
echo "Building ${TARGET}"
echo "$SEP"

mkdir -p build && cd build || exit

cmake ..

make -j8 || exit

cd ..

echo "$SEP"
echo "Running ${TARGET}"
echo "$SEP"

CMD=./build/${TARGET}

$CMD

echo "$SEP"
echo "Profiling ${TARGET}"
echo "$SEP"

nsys profile \
    --force-overwrite true \
    -o ${TARGET}.nsys-rep \
    --capture-range=cudaProfilerApi \
    $CMD

echo "$SEP"
echo "Done"
echo "$SEP"

