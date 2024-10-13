
import {chunkGridSize, voxelGridSize, toIndex1D, toIndex3D} from "./common.js";
import {doDownsampling} from "./downsampling.js";
import {doChunking} from "./chunking.js";
import {Matrix4} from "potree";


export async function generateVoxelsCompute(renderer, node){

	console.time("generate voxels");


	potree.onUpdate( () => {
		let positions = node.geometry.buffers.find(b => b.name === "position").buffer;
		// let colors = node.geometry.buffers.find(b => b.name === "color").buffer;
		let uvs = node.geometry.buffers.find(b => b.name === "uv").buffer;
		let indices = node.geometry.indices;

		let s = 1;
		let world = new Matrix4().set(
			s, 0, 0, 0,
			0, s, 0, 0,
			0, 0, s, 0,
			0, 0, 0, 1,
		);
		potree.renderer.drawMesh({positions, uvs, indices, world, image: node.material.image});

		{
			let cube = node.boundingBox.cube();
			let position = cube.center();
			let size = cube.size();
			let color = new Vector3(255, 0, 0);
			potree.renderer.drawBoundingBox(position, size, color);
		}
	});

	// doDownsampling(renderer, node);

	// let meshes = await doChunking(renderer, node);
	// for(let mesh of meshes){
	// 	doDownsampling(renderer, mesh);
	// }

	// for(let mesh of meshes){
	// 	potree.onUpdate( () => {
	// 		let positions = mesh.geometry.buffers[0].buffer;
	// 		let colors = mesh.geometry.buffers[1].buffer;
	// 		potree.renderer.drawMesh({positions, colors});
	// 	});
	// }



	// doDownsampling(renderer, node);


}