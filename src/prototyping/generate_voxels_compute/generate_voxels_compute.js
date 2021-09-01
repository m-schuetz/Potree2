
import {chunkGridSize, voxelGridSize, toIndex1D, toIndex3D} from "./common.js";
import {doDownsampling} from "./downsampling.js";
import {doChunking} from "./chunking.js";


export async function generateVoxelsCompute(renderer, node){

	console.time("generate voxels");

	let meshes = await doChunking(renderer, node);

	// doDownsampling(renderer, node);
	doDownsampling(renderer, meshes[0]);
	doDownsampling(renderer, meshes[1]);

	// for(let mesh of meshes){
	// 	potree.onUpdate( () => {
	// 		let positions = mesh.geometry.buffers[0].buffer;
	// 		let colors = mesh.geometry.buffers[1].buffer;
	// 		potree.renderer.drawMesh({positions, colors});
	// 	});
	// }



	// doDownsampling(renderer, node);


}