
import {Box3, Vector3} from "potree";

const gridSize = 128;
const numVoxels = gridSize ** 3;

function toVoxelCoord(voxelIndex, out){
	out.x = voxelIndex % gridSize;
	out.y = Math.floor((voxelIndex % (gridSize * gridSize)) / gridSize);
	out.z = Math.floor(voxelIndex / (gridSize * gridSize));
}

class Node{

	constructor(name, boundingBox){

		let numVoxels = gridSize ** 3;

		this.boundingBox = boundingBox;
		this.grid = new Float32Array(4 * numVoxels);
		this.cubeSize = boundingBox.max.x - boundingBox.min.x;
	}

	add(x, y, z, u, v){

		let ux = (x - this.boundingBox.min.x) / this.cubeSize;
		let uy = (y - this.boundingBox.min.y) / this.cubeSize;
		let uz = (z - this.boundingBox.min.z) / this.cubeSize;

		let ix = Math.floor(Math.min(gridSize * ux, gridSize - 1));
		let iy = Math.floor(Math.min(gridSize * uy, gridSize - 1));
		let iz = Math.floor(Math.min(gridSize * uz, gridSize - 1));

		let voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

		this.grid[4 * voxelIndex + 0] += u;
		this.grid[4 * voxelIndex + 1] += v;
		this.grid[4 * voxelIndex + 2] += 0;
		this.grid[4 * voxelIndex + 3] += 1;
	}

};

class VoxelBuilder{

	constructor(){
		
	}

	static build(batches){

		console.log(batches);

		let boundingBox = new Box3();

		for(let batch of batches){
			boundingBox.expandByBox(batch.geometry.boundingBox);
		}

		potree.onUpdate(() => {
			potree.renderer.drawBoundingBox(
				boundingBox.center(),
				boundingBox.size(),
				new Vector3(255, 255, 0),
			);
		});

		let cube = boundingBox.cube();
		let cubeSize = cube.size().x;
		let voxelSize = cubeSize / gridSize;

		let root = new Node("r", cube);

		for(let batch of batches){
			let position = batch.geometry.buffers.find(b => b.name === "position").buffer;
			let uv = batch.geometry.buffers.find(b => b.name === "uv").buffer;
			
			let numVertices = position.length / 3;

			for(let i = 0; i < numVertices; i++){
				let x = position[3 * i + 0];
				let y = position[3 * i + 1];
				let z = position[3 * i + 2];

				let u = uv[2 * i + 0];
				let v = uv[2 * i + 1];

				root.add(x, y, z, u, v);
			}

		}

		let voxelCoord = new Vector3();
		let voxels = [];
		for(let voxelIndex = 0; voxelIndex < numVoxels; voxelIndex++){
			let a = root.grid[4 * voxelIndex + 3];
			let r = 255 * root.grid[4 * voxelIndex + 0] / a;
			let g = 255 * root.grid[4 * voxelIndex + 1] / a;
			let b = 255 * root.grid[4 * voxelIndex + 2] / a;


			if(a > 0){
				toVoxelCoord(voxelIndex, voxelCoord);
				let position = new Vector3(
					cubeSize * (voxelCoord.x / gridSize) + cube.min.x,
					cubeSize * (voxelCoord.y / gridSize) + cube.min.y,
					cubeSize * (voxelCoord.z / gridSize) + cube.min.z,
				);
				let color = new Vector3(r, g, b);
				let size = new Vector3(voxelSize, voxelSize, voxelSize);

				voxels.push({
					voxelIndex, voxelCoord, position, color, size
				});
			}
		}

		potree.onUpdate(() => {
			for(let voxel of voxels){
				potree.renderer.drawBox(
					voxel.position,
					voxel.size,
					voxel.color,
				);
			}
		});

		console.log(voxels.length);
		console.log(boundingBox);
	}

}


export class GlbVoxelLoader{

	constructor(){

	}

	static load(url, callback){
		console.log("abc", url);

		let workerPath = "./src/misc/GLBLoaderWorker.js";
		let worker = new Worker(workerPath, {type: "module"});

		let batches = [];
		let images = new Map();

		let image_loaded = (e) => {
			images.set(e.data.imageRef, e.data.imageBitmap);
		};

		let mesh_batch_loaded = (e) => {
			batches.push({
				geometry: e.data.geometry,
				image: images.get(e.data.imageRef),
			});
		};

		let onLoaded = (e) => {
			VoxelBuilder.build(batches);
		};

		worker.onmessage = (e) => {

			if(e.data.type === "mesh_batch_loaded"){
				mesh_batch_loaded(e);
			}else if(e.data.type === "image_loaded"){
				image_loaded(e);
			}else if(e.data.type === "finished"){
				onLoaded(e);
			}

		};

		let absoluteUrl = new URL(url, document.baseURI).href;
		worker.postMessage({url: absoluteUrl});


	}

};
