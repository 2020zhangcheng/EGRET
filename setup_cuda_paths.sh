cat > ~/setup_cuda_paths.sh <<'EOF'
# 必须在激活 3dgrounding 环境后 source 本文件

if [ -z "$CONDA_PREFIX" ]; then
    echo "错误：请先激活 conda 环境"
    return 1
fi

SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
CUDA_INCLUDE="$CONDA_PREFIX/targets/x86_64-linux/include"
CUDA_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"

# 自动收集 pip 安装的 NVIDIA CUDA 组件目录
NVIDIA_INCLUDE=""
NVIDIA_LIB=""

for dir in "$SITE_PACKAGES"/nvidia/*/include; do
    [ -d "$dir" ] || continue

    # 排除已发现的 cu13 目录
    case "$dir" in
        */nvidia/cu13/*) continue ;;
    esac

    NVIDIA_INCLUDE="${NVIDIA_INCLUDE:+$NVIDIA_INCLUDE:}$dir"
done

for dir in "$SITE_PACKAGES"/nvidia/*/lib; do
    [ -d "$dir" ] || continue

    case "$dir" in
        */nvidia/cu13/*) continue ;;
    esac

    NVIDIA_LIB="${NVIDIA_LIB:+$NVIDIA_LIB:}$dir"
done

export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export TORCH_CUDA_ARCH_LIST="8.9"

# 编译器头文件搜索路径
export CPATH="$CUDA_INCLUDE${NVIDIA_INCLUDE:+:$NVIDIA_INCLUDE}${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_INCLUDE${NVIDIA_INCLUDE:+:$NVIDIA_INCLUDE}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

# 编译链接时搜索库文件，不修改运行时 LD_LIBRARY_PATH
export LIBRARY_PATH="$CUDA_LIB${NVIDIA_LIB:+:$NVIDIA_LIB}${LIBRARY_PATH:+:$LIBRARY_PATH}"

echo "CUDA 编译环境已配置"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA include dirs:"
echo "$CPATH" | tr ':' '\n'
echo "CUDA library dirs:"
echo "$LIBRARY_PATH" | tr ':' '\n'
EOF
