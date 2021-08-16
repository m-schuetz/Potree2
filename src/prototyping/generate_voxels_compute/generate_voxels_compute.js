
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


	{
		let chunk = result_chunking.chunks[322];

		let min = cube.min;
		let max = cube.max;
		let size = cube.max.x - cube.min.x;
		let coord = chunk.coord;
		let chunkSize = size / (2 ** chunk.level);


		{
		

		potree.onUpdate( () => {
			// for(let chunk of leafNodes){

				
				let x = size * (coord.x / chunkGridSize) + min.x + 0.5 * chunkSize;
				let y = size * (coord.y / chunkGridSize) + min.y + 0.5 * chunkSize;
				let z = size * (coord.z / chunkGridSize) + min.z + 0.5 * chunkSize;

				let position = new Vector3(x, y, z); //.applyMatrix4(node.world);
				let scale = new Vector3(chunkSize, chunkSize, chunkSize);
				let color = new Vector3(255, 0, 0);
				potree.renderer.drawBoundingBox(position, scale, color);

			// }
		});

		// {
		// 	// chunk.triangleOffset
		// 	// chunk.numTriangles
		// 	let indices = new Int32Array(result_chunking.rSortedIndices, 
		// 		3 * 4 * chunk.triangleOffset, 3 * chunk.numTriangles);

		// 	potree.onUpdate( () => {

		// 		let positions = node.geometry.findBuffer("position");
		// 		let colors = node.geometry.findBuffer("color");
		// 		let uvs = node.geometry.findBuffer("uv");

		// 		potree.renderer.drawMesh({
		// 			positions, colors, uvs, indices, 
		// 			image: node.material.image,
		// 			// world: node.world,

		// 		});
		// 	});

		// 	console.timeEnd("generate voxels");
		// }


	}
	}












	// {
	// 	let min = cube.min;
	// 	let max = cube.max;
	// 	let size = cube.max.x - cube.min.x;

	// 	potree.onUpdate( () => {
	// 		for(let chunk of leafNodes){

	// 			let coord = chunk.coord;
	// 			let chunkSize = size / (2 ** chunk.level);
	// 			let x = size * (coord.x / gridSize) + min.x + 0.5 * chunkSize;
	// 			let y = size * (coord.y / gridSize) + min.y + 0.5 * chunkSize;
	// 			let z = size * (coord.z / gridSize) + min.z + 0.5 * chunkSize;

	// 			let position = new Vector3(x, y, z); //.applyMatrix4(node.world);
	// 			let scale = new Vector3(chunkSize, chunkSize, chunkSize);
	// 			let color = new Vector3(255, 0, 0);
	// 			potree.renderer.drawBoundingBox(position, scale, color);

	// 		}
	// 	});
	// }

	// {

	// 	let indices = new Int32Array(rSortedIndices, 0, 3 * numTriangles);

	// 	potree.onUpdate( () => {

	// 		let positions = node.geometry.findBuffer("position");
	// 		let colors = node.geometry.findBuffer("color");
	// 		let uvs = node.geometry.findBuffer("uv");

	// 		potree.renderer.drawMesh({
	// 			positions, colors, uvs, indices, 
	// 			image: node.material.image,
	// 			// world: node.world,

	// 		});
	// 	});

	// 	console.timeEnd("generate voxels");
	// }

}