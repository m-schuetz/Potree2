
import {chunkGridSize, voxelGridSize, toIndex1D, toIndex3D} from "./common.js";
import {doDownsampling} from "./downsampling.js";
import {doChunking} from "./chunking.js";


export async function generateVoxelsCompute(renderer, node){

	console.time("generate voxels");

	let meshes = await doChunking(renderer, node);

	// potree.onUpdate( () => {
	// 	let positions = node.geometry.buffers.find(b => b.name === "position").buffer;
	// 	let colors = node.geometry.buffers.find(b => b.name === "color").buffer;
	// 	let indices = node.geometry.indices;
	// 	potree.renderer.drawMesh({positions, colors, indices});
	// });

	// doDownsampling(renderer, meshes[1]);

	for(let mesh of meshes){
		doDownsampling(renderer, mesh);
	}

	// potree.onUpdate( () => {
	// 	// {
	// 	// 	let mesh = {positions, colors};
	// 	// 	potree.renderer.drawMesh(mesh);
	// 	// }

	// 	// {
	// 	// 	let position = cube.center();
	// 	// 	let size = cube.size();
	// 	// 	let color = new Vector3(255, 0, 0);
	// 	// 	potree.renderer.drawBoundingBox(position, size, color);
	// 	// }

	// 	{
	// 		let chunkPos = node.boundingBox.center();
	// 		let chunkSize = node.boundingBox.size();
	// 		let color = new Vector3(0, 255, 0);
	// 		potree.renderer.drawBoundingBox(chunkPos, chunkSize, color);
	// 	}
	// });

	// for(let mesh of meshes){
	// 	potree.onUpdate( () => {
	// 		let positions = mesh.geometry.buffers[0].buffer;
	// 		let colors = mesh.geometry.buffers[1].buffer;
	// 		potree.renderer.drawMesh({positions, colors});
	// 	});
	// }



	// doDownsampling(renderer, node);


}