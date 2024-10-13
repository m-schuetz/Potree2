
import {Box3, Vector3} from "potree";

const gridSize = 128;
const numVoxels = gridSize ** 3;
const MAX_TRIANGLES_PER_NODE = 100_000;

let tmpCanvas = null;
let tmpContext = null;
function getTmpContext(width, height){

	if(tmpCanvas === null){
		tmpCanvas = document.createElement('canvas');
		tmpCanvas.width = width;
		tmpCanvas.height = height;
		tmpContext = tmpCanvas.getContext('2d');
	}

	return tmpContext;
}

function createChildAABB(aabb, index){
	let min = aabb.min.clone();
	let max = aabb.max.clone();
	let size = max.clone().sub(min);

	if ((index & 0b0001) > 0) {
		min.z += size.z / 2;
	} else {
		max.z -= size.z / 2;
	}

	if ((index & 0b0010) > 0) {
		min.y += size.y / 2;
	} else {
		max.y -= size.y / 2;
	}
	
	if ((index & 0b0100) > 0) {
		min.x += size.x / 2;
	} else {
		max.x -= size.x / 2;
	}

	return new Box3(min, max);
}

function toVoxelCoord(voxelIndex, out){
	out.x = voxelIndex % gridSize;
	out.y = Math.floor((voxelIndex % (gridSize * gridSize)) / gridSize);
	out.z = Math.floor(voxelIndex / (gridSize * gridSize));
}

function toVoxelIndex(x, y, z, min, gridSize, cubeSize){
	let ux = (x - min.x) / cubeSize;
	let uy = (y - min.y) / cubeSize;
	let uz = (z - min.z) / cubeSize;

	let ix = Math.floor(Math.min(gridSize * ux, gridSize - 1));
	let iy = Math.floor(Math.min(gridSize * uy, gridSize - 1));
	let iz = Math.floor(Math.min(gridSize * uz, gridSize - 1));

	let voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

	return voxelIndex;
}

class Node{

	constructor(name, boundingBox){

		let numVoxels = gridSize ** 3;

		this.boundingBox = boundingBox;
		this.children = new Array(8).fill(null);
		this.grid = new Float32Array(4 * numVoxels);
		this.cubeSize = boundingBox.max.x - boundingBox.min.x;
		this.voxels = [];
		this.childTriangleCounters = new Uint32Array(8);
		this.batches = [];
	}

	addPoint(x, y, z, r, g, b){

		let voxelIndex = toVoxelIndex(x, y, z, this.boundingBox.min, gridSize, this.cubeSize);

		this.grid[4 * voxelIndex + 0] += r;
		this.grid[4 * voxelIndex + 1] += g;
		this.grid[4 * voxelIndex + 2] += b;
		this.grid[4 * voxelIndex + 3] += 1;
	}

	addBatch(batch){
		let position = batch.geometry.buffers.find(b => b.name === "position").buffer;
		let uv = batch.geometry.buffers.find(b => b.name === "uv").buffer;
		
		let numVertices = position.length / 3;

		let {imageBitmap, imageData} = batch;
		let {width, height} = imageBitmap;

		let child_index_0 = -1;
		let child_index_1 = -1;
		for(let i = 0; i < numVertices; i++){
			let x = position[3 * i + 0];
			let y = position[3 * i + 1];
			let z = position[3 * i + 2];

			let u = uv[2 * i + 0];
			let v = uv[2 * i + 1];

			let U = Math.floor(u * width);
			let V = Math.floor(v * height);
			let pixelIndex = U + height * V;

			let r = imageData[4 * pixelIndex + 0] / 255;
			let g = imageData[4 * pixelIndex + 1] / 255;
			let b = imageData[4 * pixelIndex + 2] / 255;
			let a = 255;

			this.addPoint(x, y, z, r, g, b);

			// count triangles in child nodes
			let childIndex = toVoxelIndex(x, y, z, this.boundingBox.min, 2, this.cubeSize);

			if((i % 3) === 0){
				this.childTriangleCounters[childIndex]++;
				child_index_0 = childIndex;
			}else if((i % 3) === 1){
				if(child_index_0 !== childIndex){
					this.childTriangleCounters[childIndex]++;
				}
				child_index_1 = childIndex;
			}else{
				if(child_index_0 !== childIndex && child_index_1 !== childIndex){
					this.childTriangleCounters[childIndex]++;
				}
				child_index_0 = -1;
				child_index_1 = -1;
			}

		}

		this.batches.push(batch);
	}

	split(){

		let childBatches = new Array(8).fill(null);

		for(let childIndex = 0; childIndex < 8; childIndex++){

			let numTriangles = this.childTriangleCounters[childIndex];

			if(numTriangles < MAX_TRIANGLES_PER_NODE){
				continue;
			}

			// create child node
			let childCube = createChildAABB(this.boundingBox, childIndex);
			let child = new Node(this.name + childIndex, childCube);
			this.children[childIndex] = child;

			// create batch from triangles in child node boundary
			let childPositions = new Float32Array(3 * numTriangles);
			let childUvs = new Float32Array(2 * numTriangles);

			let geometry = {
				buffers: [
					{name: "position", buffer: childPositions},
					{name: "uv", buffer: childUvs},
				],
				buffersByName: {
					position: childPositions,
					uv: childUvs,
				},
				numElements: 3 * numTriangles,
				boundingBox: childCube,
			};

			// HACK
			let {imageBitmap, imageData} = this.batches[0];

			let childBatch = {
				numProcessed: 0,
				geometry, imageBitmap, imageData
			};

			childBatches[childIndex] = childBatch;
		}

		let addToChildBatch = (batch_position, batch_uv, childBatchIndex, triangleIndex) => {
			let childBatch = childBatches[childBatchIndex];

			let sourcePos = batch_position;
			let sourceUV = batch_uv;
			let targetPos = childBatch.geometry.buffersByName.position;
			let targetUV = childBatch.geometry.buffersByName.uv;

			let sourceIndex = triangleIndex;
			let targetIndex = childBatch.numProcessed;

			targetPos.buffer[9 * targetIndex + 0] = sourcePos[9 * sourceIndex + 0];
			targetPos.buffer[9 * targetIndex + 1] = sourcePos[9 * sourceIndex + 1];
			targetPos.buffer[9 * targetIndex + 2] = sourcePos[9 * sourceIndex + 2];
			targetPos.buffer[9 * targetIndex + 3] = sourcePos[9 * sourceIndex + 3];
			targetPos.buffer[9 * targetIndex + 4] = sourcePos[9 * sourceIndex + 4];
			targetPos.buffer[9 * targetIndex + 5] = sourcePos[9 * sourceIndex + 5];
			targetPos.buffer[9 * targetIndex + 6] = sourcePos[9 * sourceIndex + 6];
			targetPos.buffer[9 * targetIndex + 7] = sourcePos[9 * sourceIndex + 7];
			targetPos.buffer[9 * targetIndex + 8] = sourcePos[9 * sourceIndex + 8];

			targetUV.buffer[6 * targetIndex + 0] = sourceUV[6 * sourceIndex + 0];
			targetUV.buffer[6 * targetIndex + 1] = sourceUV[6 * sourceIndex + 1];
			targetUV.buffer[6 * targetIndex + 2] = sourceUV[6 * sourceIndex + 2];
			targetUV.buffer[6 * targetIndex + 3] = sourceUV[6 * sourceIndex + 3];
			targetUV.buffer[6 * targetIndex + 4] = sourceUV[6 * sourceIndex + 4];
			targetUV.buffer[6 * targetIndex + 5] = sourceUV[6 * sourceIndex + 5];

			childBatch.numProcessed++;

		};

		for(let batch of this.batches){

			let position = batch.geometry.buffers.find(b => b.name === "position").buffer;
			let uv = batch.geometry.buffers.find(b => b.name === "uv").buffer;
			let numVertices = position.length / 3;

			let child_index_0 = -1;
			let child_index_1 = -1;
			for(let i = 0; i < numVertices; i++){
				let x = position[3 * i + 0];
				let y = position[3 * i + 1];
				let z = position[3 * i + 2];

				let childIndex = toVoxelIndex(x, y, z, this.boundingBox.min, 2, this.cubeSize);
				let triangleIndex = Math.floor(i / 3);

				if(!childBatches[childIndex]){
					continue;
				}

				if((i % 3) === 0){
					addToChildBatch(position, uv, childIndex, triangleIndex);

					child_index_0 = childIndex;
				}else if((i % 3) === 1){
					if(child_index_0 !== childIndex){
						addToChildBatch(position, uv, childIndex, triangleIndex);
					}

					child_index_1 = childIndex;
				}else{
					if(child_index_0 !== childIndex && child_index_1 !== childIndex){
						addToChildBatch(position, uv, childIndex, triangleIndex);
					}

					child_index_0 = -1;
					child_index_1 = -1;
				}
			}


		}




		

	}

	finalize(){

		let cube = this.boundingBox;
		let cubeSize = cube.size().x;
		let voxelSize = cubeSize / gridSize;

		let voxelCoord = new Vector3();

		for(let voxelIndex = 0; voxelIndex < numVoxels; voxelIndex++){
			let a = this.grid[4 * voxelIndex + 3];
			let r = 255 * this.grid[4 * voxelIndex + 0] / a;
			let g = 255 * this.grid[4 * voxelIndex + 1] / a;
			let b = 255 * this.grid[4 * voxelIndex + 2] / a;


			if(a > 0){
				toVoxelCoord(voxelIndex, voxelCoord);
				let position = new Vector3(
					cubeSize * (voxelCoord.x / gridSize) + cube.min.x,
					cubeSize * (voxelCoord.y / gridSize) + cube.min.y,
					cubeSize * (voxelCoord.z / gridSize) + cube.min.z,
				);
				let color = new Vector3(r, g, b);
				let size = new Vector3(voxelSize, voxelSize, voxelSize);

				this.voxels.push({
					voxelIndex, voxelCoord, position, color, size
				});
			}
		}

	}

};

class VoxelBuilder{

	constructor(){
		
	}

	static build(batches){

		let boundingBox = new Box3();

		for(let batch of batches){
			boundingBox.expandByBox(batch.geometry.boundingBox);
		}

		// potree.onUpdate(() => {
		// 	potree.renderer.drawBoundingBox(
		// 		boundingBox.center(),
		// 		boundingBox.size(),
		// 		new Vector3(255, 255, 0),
		// 	);
		// });

		let cube = boundingBox.cube();

		let root = new Node("r", cube);

		for(let batch of batches){
			root.addBatch(batch);
		}

		root.split();
		root.finalize();

		console.log(root);

		potree.onUpdate(() => {
			for(let voxel of root.voxels){
				potree.renderer.drawBox(
					voxel.position,
					voxel.size,
					voxel.color,
				);
			}
		});

		potree.onUpdate(() => {

			for(let i = 0; i < 8; i++){

				let child = root.children[i];

				if(child === null){
					continue;
				}

				let color = new Vector3(
					255 * i / 8,
					0,
					0,
				);

				potree.renderer.drawBoundingBox(
					child.boundingBox.center(),
					child.boundingBox.size(),
					color
				);
			}
		});

		// console.log(voxels.length);
		// console.log(boundingBox);
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

			let imageBitmap = e.data.imageBitmap;
			let context = getTmpContext(imageBitmap.width, imageBitmap.height);
			context.drawImage(imageBitmap, 0, 0);
			let imageData = context.getImageData(0, 0, imageBitmap.width, imageBitmap.height).data;

			// images.set(e.data.imageRef, e.data.imageBitmap);

			images.set(e.data.imageRef, {imageBitmap, imageData});
		};

		let mesh_batch_loaded = (e) => {

			let {imageBitmap, imageData} = images.get(e.data.imageRef);

			batches.push({
				geometry: e.data.geometry,
				// image: images.get(e.data.imageRef),
				imageBitmap, imageData,
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
