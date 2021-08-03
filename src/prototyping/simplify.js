
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


export function simplify(node){

	let geometry = node.geometry;
	let positions = new Float32Array(node.geometry.buffers[2].buffer.buffer);
	let indices = geometry.indices;
	let numVertices = positions.length / 3;
	let numTriangles = indices.length / 3;
	let triangleIDs = new Uint32Array(numTriangles);


	// let permutation = randomPermutation(1000, numTriangles);
	let permutation = new Uint32Array(1000);
	for(let i = 0; i < 1000; i++){
		permutation[i] = 1000 * i;
	}
	let centers = [];

	let mappings = [];

	for(let triangleIndex of permutation){

		let i0 = indices[3 * triangleIndex + 0];
		let i1 = indices[3 * triangleIndex + 1];
		let i2 = indices[3 * triangleIndex + 2];

		let p0 = new Vector3(positions[3 * i0 + 0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
		let p1 = new Vector3(positions[3 * i1 + 0], positions[3 * i1 + 1], positions[3 * i1 + 2]);
		let p2 = new Vector3(positions[3 * i2 + 0], positions[3 * i2 + 1], positions[3 * i2 + 2]);

		let center = new Vector3().add(p0).add(p1).add(p2).multiplyScalar(0.333333);
		center = p0;
		center.applyMatrix4(node.world);
		// centers.push(center);

		triangleIDs[triangleIndex] = triangleIndex;

		mappings.push(
			[i0, triangleIndex],
			[i1, triangleIndex],
			[i2, triangleIndex],
		);
	}

	let vertexNeighbors = new Array(numVertices).fill(0).map(v => []);
	
	for(let triangleIndex = 0; triangleIndex < numTriangles; triangleIndex++){
		let i0 = indices[3 * triangleIndex + 0];
		let i1 = indices[3 * triangleIndex + 1];
		let i2 = indices[3 * triangleIndex + 2];

		vertexNeighbors[i0].push(triangleIndex);
		vertexNeighbors[i1].push(triangleIndex);
		vertexNeighbors[i2].push(triangleIndex);
	}

	for(let triangleIndex of vertexNeighbors[0]){
		let i0 = indices[3 * triangleIndex + 0];
		let i1 = indices[3 * triangleIndex + 1];
		let i2 = indices[3 * triangleIndex + 2];

		let p0 = new Vector3(positions[3 * i0 + 0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
		let p1 = new Vector3(positions[3 * i1 + 0], positions[3 * i1 + 1], positions[3 * i1 + 2]);
		let p2 = new Vector3(positions[3 * i2 + 0], positions[3 * i2 + 1], positions[3 * i2 + 2]);

		let center = new Vector3().add(p0).add(p1).add(p2).multiplyScalar(0.333333);
		center = p0;
		center.applyMatrix4(node.world);
		centers.push(center);
		
	}

	// for(let triangleIndex = 0; triangleIndex < numTriangles; triangleIndex++){

	// 	if(triangleIDs[triangleIndex] > 0){

	// 		let i0 = indices[3 * triangleIndex + 0];
	// 		let i1 = indices[3 * triangleIndex + 1];
	// 		let i2 = indices[3 * triangleIndex + 2];

	// 		for(let triangleID of vertexNeighbors[i0]){
	// 			triangleIDs[triangleID] = triangleIndex;
	// 		}
	// 	}
	// }

	{

		let geometry = new Geometry();
		geometry.buffers = [
			...node.geometry.buffers,
			{name: "triangle_ids", buffer: new Uint8Array(triangleIDs.buffer)},
		];


		geometry.indices = node.geometry.indices;

		let node2 = new Mesh("wireframe", geometry);

		node2.material = new TriangleColorMaterial();
		node2.material.color.set(0, 1, 0);
		node2.position.copy(node.position);
		node2.scale.copy(node.scale);
		node2.rotation.copy(node.rotation);

		scene.root.children.push(node2);
	}


	potree.onUpdate( () => {

		for(let point of centers){
			potree.renderer.drawBox(
				point, 
				new Vector3(
					0.1, 
					0.1, 
					0.1),
				new Vector3(255, 0, 0),
			);
		}
	});



}