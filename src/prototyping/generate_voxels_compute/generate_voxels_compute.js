
import {chunkGridSize, voxelGridSize, toIndex1D, toIndex3D} from "./common.js";
import {doDownsampling} from "./downsampling.js";
import {doChunking} from "./chunking.js";


export async function generateVoxelsCompute(renderer, node){

	console.time("generate voxels");

	doChunking(renderer, node);

	// doDownsampling(renderer, node);


}