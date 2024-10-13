
import {chunkGridSize, voxelGridSize, toIndex1D, toIndex3D} from "./common.js";
import {doDownsampling} from "./downsampling.js";


export async function generateVoxelsCompute(renderer, node){

	console.time("generate voxels");

	doDownsampling(renderer, node);


}