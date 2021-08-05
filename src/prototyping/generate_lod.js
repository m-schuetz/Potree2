
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


export function generateLOD(node){

	let geometry = node.geometry;
	let positions = new Float32Array(node.geometry.buffers[2].buffer.buffer);
	let indices = geometry.indices;
	let numVertices = positions.length / 3;
	let numTriangles = indices.length / 3;
	// let triangleIDs = new Uint32Array(numTriangles);


	let edges = [];
	let tmp0 = new Vector3();
	let tmp1 = new Vector3();

	let lengthBetween = (i0, i1) => {
		tmp0.set(positions[3 * i0 + 0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
		tmp1.set(positions[3 * i1 + 0], positions[3 * i1 + 1], positions[3 * i1 + 2]);

		return tmp0.distanceTo(tmp1);
	};

	let indexToPosition = (index) => {
		return new Vector3(positions[3 * index + 0], positions[3 * index + 1], positions[3 * index + 2]);
	};

	for(let triangleID = 0; triangleID < numTriangles; triangleID++){
		let i0 = indices[3 * triangleID + 0];
		let i1 = indices[3 * triangleID + 1];
		let i2 = indices[3 * triangleID + 2];

		let e01 = {length: lengthBetween(i0, i1), start: i0, end: i1};
		let e12 = {length: lengthBetween(i1, i2), start: i1, end: i2};
		let e20 = {length: lengthBetween(i2, i0), start: i2, end: i0};

		edges.push(e01, e12, e20);
	}

	let collapsedEdges = [];
	let threshold = 0.01;
	for(let edgeID = 0; edgeID < edges.length; edgeID++){

		let edge = edges[edgeID];

		if(edge.length < threshold){
			// collapse
			// update adjacent
			// add collapsed to visualization
			let start = indexToPosition(edge.start).applyMatrix4(node.world);
			let end = indexToPosition(edge.end).applyMatrix4(node.world);
			collapsedEdges.push([start, end]);
		}

	}



	potree.onUpdate( () => {

		let color = new Vector3(255, 0, 0);
		for(let edge of collapsedEdges){
			potree.renderer.drawLine(edge[0], edge[1], color);
		}

		// for(let point of centers){
			// potree.renderer.drawBox(
			// 	point, 
			// 	new Vector3(
			// 		0.01, 
			// 		0.01, 
			// 		0.01),
			// 	new Vector3(255, 255, 0),
			// );
		// }
	});



}