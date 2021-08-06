
import {Vector3, Box3} from "potree" 
import {Mesh, Geometry, WireframeMaterial, TriangleColorMaterial} from "potree";

class Cluster{

	constructor(){

		this.level = 0;
		this.id = 0;
		this.voxels = [];
		this.children = [];
		this.boundingBox = new Box3();
		let voxelSize = 1;
		this.color = new Vector3(
			255 * Math.random(), 
			255 * Math.random(), 
			255 * Math.random(), 
		);

	}

};

let subClusterResolution = 4;
let subVoxelResolution = 64;

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

function getVertexPos(positions, index){

	return new Vector3(
		positions[3 * index + 0], 
		positions[3 * index + 1], 
		positions[3 * index + 2]
	);

};

let normalize = (I, gridSize) => {
	return Math.min(Math.max(Math.floor(I), 0), gridSize - 1);
}


function generateChildClusters(cluster, node){

	let geometry = node.geometry;
	let positions = new Float32Array(node.geometry.buffers[2].buffer.buffer);
	let indices = geometry.indices;
	let numTriangles = indices.length / 3;
	let uvs = new Float32Array(node.geometry.buffers[3].buffer.buffer);

	let boundingBox = cluster.boundingBox;
	let boundingBoxSize = boundingBox.size();
	let cubeSize = Math.max(...boundingBox.size().toArray());
	let gridSize = subVoxelResolution;

	let clusterGridSize = subClusterResolution;
	let grid = new Uint32Array(gridSize ** 3);
	let voxelSize = cubeSize / gridSize;

	let {width, height} = node.material.image;
	let imgData = node.material.imageData.data;

	let getColor = (index) => {
		let u = uvs[2 * index + 0];
		let v = uvs[2 * index + 1];
		let px = Math.floor(u * width);
		let py = Math.floor(v * height);

		let pixelID = py * width + px;
		let r = Math.min(Math.floor(imgData[4 * pixelID + 0]), 255);
		let g = Math.min(Math.floor(imgData[4 * pixelID + 1]), 255);
		let b = Math.min(Math.floor(imgData[4 * pixelID + 2]), 255);

		return [r, g, b];
	};

	let samples = [];
	for(let i = 0; i < numTriangles; i++){

		let i0 = indices[3 * i + 0];
		let i1 = indices[3 * i + 1];
		let i2 = indices[3 * i + 2];

		let p0 = getVertexPos(positions, i0);
		let p1 = getVertexPos(positions, i1);
		let p2 = getVertexPos(positions, i2);

		let center = p0.clone().add(p1).add(p2).divideScalar(3);

		let ix = gridSize * (center.x - boundingBox.min.x) / cubeSize;
		let iy = gridSize * (center.y - boundingBox.min.y) / cubeSize;
		let iz = gridSize * (center.z - boundingBox.min.z) / cubeSize;

		let outsideX = ix < 0 || ix >= gridSize;
		let outsideY = iy < 0 || iy >= gridSize;
		let outsideZ = iz < 0 || iz >= gridSize;
		
		if(outsideX || outsideY || outsideZ){
			continue;
		}

		ix = normalize(ix, gridSize);
		iy = normalize(iy, gridSize);
		iz = normalize(iz, gridSize);

		let gridIndex = ix + gridSize * iy + gridSize * gridSize * iz;
		let oldVal = grid[gridIndex];
		grid[gridIndex]++;
		
		let [r0, g0, b0] = getColor(i0);
		let [r1, g1, b1] = getColor(i1);
		let [r2, g2, b2] = getColor(i2);
		let r = (r0 + r1 + r2) / 3;
		let g = (g0 + g1 + g2) / 3;
		let b = (b0 + b1 + b2) / 3;

		let boxCenter = new Vector3(
			(ix / gridSize) * cubeSize + boundingBox.min.x,
			(iy / gridSize) * cubeSize + boundingBox.min.y,
			(iz / gridSize) * cubeSize + boundingBox.min.z,
		);

		if(oldVal === 0){
			samples.push({
				position: boxCenter,
				worldPosition: boxCenter.clone().applyMatrix4(node.world),
				scale: new Vector3(1, 1, 1).multiplyScalar(cubeSize / gridSize),
				color: new Vector3(r, g, b),
			});
		}
	}

	let clusters = [];
	let clusterGrid = [];

	let clusterBoxSize = cubeSize / clusterGridSize;

	for(let x = 0; x < clusterGridSize; x++){
	for(let y = 0; y < clusterGridSize; y++){
	for(let z = 0; z < clusterGridSize; z++){

		let id = x + clusterGridSize * y + clusterGridSize * clusterGridSize * z;

		let min = new Vector3(
			cluster.boundingBox.min.x + clusterBoxSize * x,
			cluster.boundingBox.min.y + clusterBoxSize * y,
			cluster.boundingBox.min.z + clusterBoxSize * z,
		);
		let max = new Vector3(
			cluster.boundingBox.min.x + clusterBoxSize * (x + 1),
			cluster.boundingBox.min.y + clusterBoxSize * (y + 1),
			cluster.boundingBox.min.z + clusterBoxSize * (z + 1),
		);
		let boundingBox = new Box3(min, max);

		let childCluster = new Cluster();
		childCluster.level = cluster.level + 1;
		childCluster.boundingBox = boundingBox;
		childCluster.id = id;
		childCluster.voxelSize = voxelSize;
		clusterGrid[id] = childCluster;
	}
	}
	}

	for(let voxel of samples){

		let samplePos = voxel.position;
		let ix = normalize(clusterGridSize * (samplePos.x - boundingBox.min.x) / cubeSize, gridSize);
		let iy = normalize(clusterGridSize * (samplePos.y - boundingBox.min.y) / cubeSize, gridSize);
		let iz = normalize(clusterGridSize * (samplePos.z - boundingBox.min.z) / cubeSize, gridSize);

		let clusterGridIndex = ix + clusterGridSize * iy + clusterGridSize * clusterGridSize * iz;
		let cluster = clusterGrid[clusterGridIndex];

		cluster.voxels.push(voxel);

	}

	for(let cluster of clusterGrid){
		if(cluster.voxels.length > 0){
			clusters.push(cluster);
		}
	}


	for(let cluster of clusters){

		if(![0, 1, 2, 3, 6, 16, 17, 18].includes(cluster.id)){
			continue;
		}

		if(cluster.boundingBox.size().length() < 0.2){
			continue;
		}

		let childClusters = generateChildClusters(cluster, node);
		cluster.children = childClusters;
	}

	return clusters;
}


export function generateVoxels(node){

	let geometry = node.geometry;
	let positions = new Float32Array(node.geometry.buffers[2].buffer.buffer);
	let indices = geometry.indices;
	let numTriangles = indices.length / 3;
	let uvs = new Float32Array(node.geometry.buffers[3].buffer.buffer);

	let boundingBox = node.boundingBox;
	let cubeSize = Math.max(...boundingBox.size().toArray());
	let gridSize = subVoxelResolution;
	let clusterGridSize = subClusterResolution;
	let grid = new Uint32Array(gridSize ** 3);
	let voxelSize = cubeSize / gridSize;

	let {width, height} = node.material.image;
	let imgData = node.material.imageData.data;

	let getColor = (index) => {
		let u = uvs[2 * index + 0];
		let v = uvs[2 * index + 1];
		let px = Math.floor(u * width);
		let py = Math.floor(v * height);

		let pixelID = py * width + px;
		let r = Math.min(Math.floor(imgData[4 * pixelID + 0]), 255);
		let g = Math.min(Math.floor(imgData[4 * pixelID + 1]), 255);
		let b = Math.min(Math.floor(imgData[4 * pixelID + 2]), 255);

		return [r, g, b];
	};

	let samples = [];
	for(let i = 0; i < numTriangles; i++){

		let i0 = indices[3 * i + 0];
		let i1 = indices[3 * i + 1];
		let i2 = indices[3 * i + 2];

		let p0 = getVertexPos(positions, i0);
		let p1 = getVertexPos(positions, i1);
		let p2 = getVertexPos(positions, i2);

		let center = p0.clone().add(p1).add(p2).divideScalar(3);

		let ix = normalize(gridSize * (center.x - boundingBox.min.x) / cubeSize, gridSize);
		let iy = normalize(gridSize * (center.y - boundingBox.min.y) / cubeSize, gridSize);
		let iz = normalize(gridSize * (center.z - boundingBox.min.z) / cubeSize, gridSize);

		let gridIndex = ix + gridSize * iy + gridSize * gridSize * iz;
		let oldVal = grid[gridIndex];
		grid[gridIndex]++;

		
		let [r0, g0, b0] = getColor(i0);
		let [r1, g1, b1] = getColor(i1);
		let [r2, g2, b2] = getColor(i2);
		let r = (r0 + r1 + r2) / 3;
		let g = (g0 + g1 + g2) / 3;
		let b = (b0 + b1 + b2) / 3;

		let boxCenter = new Vector3(
			(ix / gridSize) * cubeSize + boundingBox.min.x,
			(iy / gridSize) * cubeSize + boundingBox.min.y,
			(iz / gridSize) * cubeSize + boundingBox.min.z,
		);

		if(oldVal === 0){
			samples.push({
				position: boxCenter,
				worldPosition: boxCenter.clone().applyMatrix4(node.world),
				scale: new Vector3(1, 1, 1).multiplyScalar(cubeSize / gridSize),
				color: new Vector3(r, g, b),
			});
		}
	}


	let clusters = [];
	let clusterGrid = [];

	let clusterBoxSize = cubeSize / clusterGridSize;

	for(let x = 0; x < clusterGridSize; x++){
	for(let y = 0; y < clusterGridSize; y++){
	for(let z = 0; z < clusterGridSize; z++){

		let id = x + clusterGridSize * y + clusterGridSize * clusterGridSize * z;

		let min = new Vector3(
			node.boundingBox.min.x + clusterBoxSize * x,
			node.boundingBox.min.y + clusterBoxSize * y,
			node.boundingBox.min.z + clusterBoxSize * z,
		);
		let max = new Vector3(
			node.boundingBox.min.x + clusterBoxSize * (x + 1),
			node.boundingBox.min.y + clusterBoxSize * (y + 1),
			node.boundingBox.min.z + clusterBoxSize * (z + 1),
		);
		let boundingBox = new Box3(min, max);

		let cluster = new Cluster();
		cluster.level = 0;
		cluster.boundingBox = boundingBox;
		cluster.id = id;
		cluster.voxelSize = voxelSize;
		clusterGrid[id] = cluster;
	}
	}
	}


	for(let voxel of samples){

		let samplePos = voxel.position;
		let ix = normalize(clusterGridSize * (samplePos.x - boundingBox.min.x) / cubeSize, gridSize);
		let iy = normalize(clusterGridSize * (samplePos.y - boundingBox.min.y) / cubeSize, gridSize);
		let iz = normalize(clusterGridSize * (samplePos.z - boundingBox.min.z) / cubeSize, gridSize);

		let clusterGridIndex = ix + clusterGridSize * iy + clusterGridSize * clusterGridSize * iz;
		let cluster = clusterGrid[clusterGridIndex];

		cluster.voxels.push(voxel);
	}


	for(let cluster of clusterGrid){

		// let isSelected = [18].includes(cluster.id);
		// let isSelected = [18, 22].includes(cluster.id);
		let isSelected = true;
		if(cluster.voxels.length > 0 && isSelected){
			
			clusters.push(cluster);
		}
	}

	for(let cluster of clusters){
		let childClusters = generateChildClusters(cluster, node);
		cluster.children = childClusters;
	}
	


	window.node = node;

	potree.onUpdate( () => {

		// for(let cluster of expandedClusters){

		// 	for(let voxel of cluster.voxels){
		// 		potree.renderer.drawBox(
		// 			voxel.worldPosition, voxel.scale, cluster.color
		// 		);
		// 	}

		// 	potree.renderer.drawBoundingBox(
		// 		cluster.boundingBox.center().applyMatrix4(node.world),
		// 		cluster.boundingBox.size().multiplyScalar(node.scale.x),
		// 		new Vector3(255, 0, 0),
		// 	);
		// }

		// for(let cluster of clusters){
		// 	potree.renderer.drawBoundingBox(
		// 		cluster.boundingBox.center().applyMatrix4(node.world),
		// 		cluster.boundingBox.size().multiplyScalar(node.scale.x),
		// 		new Vector3(0, 255, 0),
		// 	);
		// }

		let selectedClusters = [];

		let traverse = (cluster) => {

			let center = cluster.boundingBox.center();
			let size = cluster.boundingBox.size().length();
			let camWorldPos = camera.getWorldPosition();
			let distance = camWorldPos.distanceTo(center);

			let expand = (size / distance) > 0.5;

			if(expand && cluster.children.length > 0){
				for(let child of cluster.children){
					traverse(child);
				}
			}else{
				selectedClusters.push(cluster);
			}

		};

		for(let cluster of clusters){
			traverse(cluster);
		}

		// for(let cluster of clusters){

		// 	let center = cluster.boundingBox.center();
		// 	let size = cluster.boundingBox.size().length();
		// 	let camWorldPos = camera.getWorldPosition();
		// 	let distance = camWorldPos.distanceTo(center);

		// 	let expand = (size / distance) > 0.5;
		// 	let color = expand ? new Vector3(0, 255, 0) : new Vector3(0, 0, 255);

		// 	if(expand && cluster.children.length > 0){
		// 		for(let child of cluster.children){
		// 			potree.renderer.drawBoundingBox(
		// 				child.boundingBox.center().applyMatrix4(node.world),
		// 				child.boundingBox.size().multiplyScalar(node.scale.x),
		// 				new Vector3(255, 0, 0),
		// 			);
		// 		}
		// 		selectedClusters.push(...cluster.children);
		// 	}else{
		// 		selectedClusters.push(cluster);
		// 		potree.renderer.drawBoundingBox(
		// 			cluster.boundingBox.center().applyMatrix4(node.world),
		// 			cluster.boundingBox.size().multiplyScalar(node.scale.x),
		// 			color,
		// 		);
		// 	}
		// }

		for(let cluster of selectedClusters){
			for(let voxel of cluster.voxels){
				potree.renderer.drawBox(
					voxel.worldPosition, voxel.scale, voxel.color
				);
			}

			let color = new Vector3();
			if(cluster.level == 0){
				color.set(43,131,186);
			}else if(cluster.level == 1){
				color.set(171,221,164);
			}else if(cluster.level == 2){
				color.set(253,174,97);
			}

			potree.renderer.drawBoundingBox(
				cluster.boundingBox.center().applyMatrix4(node.world),
				cluster.boundingBox.size(),
				color,
			);
		}



	});



}