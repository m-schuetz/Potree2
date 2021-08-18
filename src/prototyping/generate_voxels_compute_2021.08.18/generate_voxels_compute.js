
import {chunkGridSize, voxelGridSize, toIndex1D, toIndex3D} from "./common.js";
import {storage_flags, uniform_flags} from "./common.js";
import {doChunking} from "./chunking.js";
import {doDownsampling} from "./downsampling.js";


export async function generateVoxelsCompute(renderer, node){

	let {device} = renderer;

	let numTriangles = node.geometry.indices.length / 3;
	let box = node.boundingBox.clone();
	let cube = box.cube();

	console.time("generate voxels");

	let result_chunking = await doChunking(renderer, node);
	let result_downsampling = await doDownsampling(renderer, node, result_chunking);


}