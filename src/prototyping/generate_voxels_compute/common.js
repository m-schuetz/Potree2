
export let chunkGridSize = 4;
export let voxelGridSize = 32;

export function toIndex1D(gridSize, voxelPos){

	return voxelPos.x 
		+ gridSize * voxelPos.y 
		+ gridSize * gridSize * voxelPos.z;
}

export function toIndex3D(gridSize, voxelIndex){
	let z = Math.floor(voxelIndex / (gridSize * gridSize));
	let y = Math.floor((voxelIndex - gridSize * gridSize * z) / gridSize);
	let x = voxelIndex % gridSize;

	return new Vector3(x, y, z);
}

export let storage_flags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
export let uniform_flags = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;