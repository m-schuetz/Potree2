
import {Vector3, Mesh, Geometry, WireframeMaterial, TriangleColorMaterial} from "potree";

function randomPermutation(n, max){

	let i64 = new BigInt64Array(max);
	let i32 = new Int32Array(i64.buffer);

	for(let i = 0; i < max; i++){
		// most significant bits for shuffle value
		i32[2 * i + 1] = Math.random() * 100_000_000;
		// least significant bits for index value
		i32[2 * i + 0] = i;
	}

	i64.sort();

	let result = new Uint32Array(n);
	for(let i = 0; i < n; i++){
		result[i] = i32[2 * i + 0];
	}

	return result;
}


export function generateVoxels(node){

	let geometry = node.geometry;
	let positions = new Float32Array(node.geometry.buffers[2].buffer.buffer);
	let indices = geometry.indices;
	let numVertices = positions.length / 3;
	let numTriangles = indices.length / 3;
	let uvs = new Float32Array(node.geometry.buffers[3].buffer.buffer);
	// let triangleIDs = new Uint32Array(numTriangles);

	let boundingBox = node.boundingBox;
	let cubeSize = Math.max(...boundingBox.size().toArray());
	let gridSize = 128;
	let grid = new Uint32Array(gridSize ** 3);

	let getVertexPos = (index) => {

		return new Vector3(
			positions[3 * index + 0], 
			positions[3 * index + 1], 
			positions[3 * index + 2]
		);
	};

	let normalize = (I) => {
		return Math.min(Math.max(Math.floor(I), 0), gridSize - 1);
	}

	let samples = [];
	for(let i = 0; i < numTriangles; i++){

		let i0 = indices[3 * i + 0];
		let i1 = indices[3 * i + 1];
		let i2 = indices[3 * i + 2];

		let p0 = getVertexPos(i0);
		let p1 = getVertexPos(i1);
		let p2 = getVertexPos(i2);

		let center = p0.clone().add(p1).add(p2).divideScalar(3);

		let ix = normalize(gridSize * (center.x - boundingBox.min.x) / cubeSize);
		let iy = normalize(gridSize * (center.y - boundingBox.min.y) / cubeSize);
		let iz = normalize(gridSize * (center.z - boundingBox.min.z) / cubeSize);

		let gridIndex = ix + gridSize * iy + gridSize * gridSize * iz;
		let oldVal = grid[gridIndex];
		grid[gridIndex]++;

		let {width, height} = node.material.image;
		let imgData = node.material.imageData.data;
		let u = uvs[2 * i2 + 0];
		let v = uvs[2 * i2 + 1];
		let px = Math.floor(u * width);
		let py = Math.floor(v * height);
		let r = imgData[4 * (height - py - 1) * width + 4 * px + 0];
		let g = imgData[4 * (height - py - 1) * width + 4 * px + 1];
		let b = imgData[4 * (height - py - 1) * width + 4 * px + 2];


		let boxCenter = new Vector3(
			(ix / gridSize) * cubeSize + boundingBox.min.x,
			(iy / gridSize) * cubeSize + boundingBox.min.y,
			(iz / gridSize) * cubeSize + boundingBox.min.z,
		);

		if(oldVal === 0){
			samples.push({
				position: boxCenter.applyMatrix4(node.world),
				scale: new Vector3(1, 1, 1).multiplyScalar(4 * cubeSize / gridSize),
				// color: new Vector3(255 * u, 255 * v, 0),
				// color: new Vector3(255 * px / width, 255 * py / height, 0),
				color: new Vector3(255 * r, 255 * g, 255 * b),
			});
		}


	}

	window.node = node;


	potree.onUpdate( () => {

		// let scale = new Vector3(0.1, 0.1, 0.1);
		let color = new Vector3(255, 0, 0);
		for(let sample of samples){
			potree.renderer.drawBox(
				sample.position, sample.scale, sample.color
			);
		}

	});



}