RED='\e[1;31m'
NC='\e[0m' # No Color or other format

make CUDA_ARCH_FLAGS=-arch=sm_35

function cublastest_on_GPU
{
	GPUnum=$1
	CUDA_VISIBLE_DEVICES=$GPUnum ./hgemm &> cublastest.out &
	P1=$!
	nvidia-smi -i $GPUnum --query-gpu=timestamp,index,name,pcie.link.gen.current,pcie.link.gen.max,pstate,clocks.current.graphics,clocks.max.graphics --format=csv -l 5 &
	P2=$!
	# echo -e "${RED}INFO:${NC} waiting for $P1 and then kill $P2"
	wait $P1 
	kill -9 $P2
	wait $! 2>/dev/null
}

mapfile -t lines < <(nvidia-smi topo -m | grep "^GPU[0-9]\+")
_SIZE=${#lines[@]} # shows the amount of available GPUs

echo -e "${RED}INFO:${NC} Running test for totally $_SIZE deivce(s) on host $(hostname)"
for (( index=0; index<$_SIZE; index++ ))
do
	echo -e "${RED}INFO:${NC} testing GPU$index"
	cublastest_on_GPU $index
done

