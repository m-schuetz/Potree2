
import {Box3} from "potree";

export let chunkGridSize = 2;
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

export function computeChildBox(parentBox, childIndex){

	let center = parentBox.center();
	let box = new Box3();

	if((childIndex & 0b001) === 0){
		box.min.x = parentBox.min.x;
		box.max.x = center.x;
	}else{
		box.min.x = center.x;
		box.max.x = parentBox.max.x;
	}

	if((childIndex & 0b010) === 0){
		box.min.y = parentBox.min.y;
		box.max.y = center.y;
	}else{
		box.min.y = center.y;
		box.max.y = parentBox.max.y;
	}

	if((childIndex & 0b100) === 0){
		box.min.z = parentBox.min.z;
		box.max.z = center.z;
	}else{
		box.min.z = center.z;
		box.max.z = parentBox.max.z;
	}

	return box;
}

export let storage_flags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
export let uniform_flags = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export class Chunk{

	constructor(){
		this.voxelOffset = 0;
		this.numVoxels = 0;
		this.triangleOffset = 0;
		this.numTriangles = 0;
		this.boundingBox = new Box3();
		this.level = 0;

		this.children = [];
	}

};
