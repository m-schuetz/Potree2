
// import {Vector3, Box3} from "potree";
import {Vector3, Box3} from "../math/math.js";

async function loadMesh(mesh, json, bufferViews, meshIndex){

	let numIndices = 0;
	for(let primitive of mesh.primitives){
		if(typeof primitive.indices !== "undefined"){
			let indices_accessor = json.accessors[primitive.indices];

			numIndices += indices_accessor.count;
		}else{
			// skip for now
		}
	}
	let numTriangles = numIndices / 3;
	let numTrianglesProcessed = 0;

	let positions = new Float32Array(9 * numTriangles);
	let uvs = new Float32Array(6 * numTriangles);
	let indices = new Uint32Array(numIndices);

	let geometry = {
		buffers: [
			{name: "position", buffer: positions},
			{name: "uv", buffer: uvs},
		],
		indices: indices,
		numElement: 0,
	};

	for(let primitive of mesh.primitives){

		if(typeof primitive.indices === "undefined"){
			continue;
		}

		let prim_indices; 
		{
			let accessor = json.accessors[primitive.indices];
			let bufferView = bufferViews[accessor.bufferView];

			if(accessor.componentType == 5123){
				prim_indices = new Uint32Array(new Uint16Array(bufferView.buffer))
			}else{
				prim_indices = new Uint32Array(bufferView.buffer);
			}
		}

		let numTriangles_primitive = prim_indices.length / 3;

		if(typeof primitive.attributes.POSITION !== "undefined")
		{
			let accessorRef = primitive.attributes.POSITION ?? null;
			let accessor = json.accessors[accessorRef];
			let bufferView = bufferViews[accessor.bufferView];
			let prim_positions = new Float32Array(bufferView.buffer)

			for(let i = 0; i < prim_indices.length; i++){
				let index = prim_indices[i];
				let x = prim_positions[3 * index + 0];
				let y = prim_positions[3 * index + 1];
				let z = prim_positions[3 * index + 2];

				positions[9 * numTrianglesProcessed + 3 * i + 0] = x;
				positions[9 * numTrianglesProcessed + 3 * i + 1] = y;
				positions[9 * numTrianglesProcessed + 3 * i + 2] = z;
			}
		}else{
			throw "wat?";
		}

		if(typeof primitive.attributes.TEXCOORD_0 !== "undefined")
		{
			let accessorRef = primitive.attributes.TEXCOORD_0 ?? null;
			let accessor = json.accessors[accessorRef];
			let bufferView = bufferViews[accessor.bufferView];
			let prim_uvs = new Float32Array(bufferView.buffer);

			for(let i = 0; i < prim_indices.length; i++){
				let index = prim_indices[i];
				let u = prim_uvs[2 * index + 0];
				let v = prim_uvs[2 * index + 1];

				uvs[6 * numTrianglesProcessed + 2 * i + 0] = u;
				uvs[6 * numTrianglesProcessed + 2 * i + 1] = v;
			}
		}else{
			for(let i = 0; i < prim_indices.length; i++){
				let u = 0;
				let v = 0;

				uvs[6 * numTrianglesProcessed + 2 * i + 0] = u;
				uvs[6 * numTrianglesProcessed + 2 * i + 1] = v;
			}
		}

		numTrianglesProcessed += numTriangles_primitive;
	}

	// BOUNDING BOX
	let boundingBox = new Box3(); {

		// let f32 = geometry.buffers.find(b => b.name === "position").buffer;

		let buffer = geometry.buffers.find(b => b.name === "position").buffer;
		let f32 = new Float32Array(buffer.buffer);

		let tmp = new Vector3();
		for(let i = 0; i < f32.length / 3; i++){
			tmp.x = f32[3 * i + 0];
			tmp.y = f32[3 * i + 1];
			tmp.z = f32[3 * i + 2];
			boundingBox.expandByPoint(tmp);
		}
	}

	geometry.boundingBox = boundingBox;

	let imageBitmap = null;
	let imageBuffer = null;
	if(json.images?.length > 0){
		let materialRef = mesh.primitives[0].material;
		let material = json.materials[materialRef];
		let imageRef = material.pbrMetallicRoughness.baseColorTexture.index;

		let image = json.images[imageRef];
		let buffer = bufferViews[image.bufferView].buffer;
		let mimeType = image.mimeType;

		var u8 = new Uint8Array(buffer);
		var blob = new Blob([u8], {type: mimeType});

		imageBitmap = await createImageBitmap(blob);
		imageBuffer = buffer;
	}

	for(let i = 0; i < indices.length; i++){
		indices[i] = i;
	}

	return {
		geometry: geometry,
		imageBitmap, imageBuffer,
	};
}

onmessage = async function(e){

	let {url} = e.data;

	let response = await fetch(url);
	let buffer = await response.arrayBuffer();
	let view = new DataView(buffer);

	let tStart = performance.now();

	let magic = view.getUint32(0, true);
	let version = view.getUint32(4, true);
	let length = view.getUint32(8, true);
	let chunkSize = view.getUint32(12, true);
	let chunkType = view.getUint32(16, true);

	let jsonBuffer = buffer.slice(20, 20 + chunkSize);
	let decoder = new TextDecoder();
	let jsonText = decoder.decode(jsonBuffer);
	let json = JSON.parse(jsonText);

	let binaryChunkOffset = 12 + 8 + chunkSize + 8;
	let binaryChunkSize = view.getUint32(20 + chunkSize, true);
	
	let bufferViews = [];
	for(let glbBufferView of json.bufferViews){

		let offset = glbBufferView.byteOffset;
		let size = glbBufferView.byteLength;

		let data = buffer.slice(binaryChunkOffset + offset, binaryChunkOffset + offset + size);

		bufferViews.push({buffer: data});
	}

	let meshIndex = 0;
	for(let glbMesh of json.meshes){

		let mesh = await loadMesh(glbMesh, json, bufferViews, meshIndex);

		if(glbMesh.primitives.length > 1){
			continue;
		}

		let message = {
			geometry: mesh.geometry,
			imageBitmap: mesh.imageBitmap,
			imageBuffer: mesh.imageBuffer,
		};

		let transferables = [
			...mesh.geometry.buffers.map(b => b.buffer.buffer),
			mesh.geometry.indices.buffer,
		];

		if(mesh.imageBitmap){
			transferables.push(mesh.imageBitmap);
			transferables.push(mesh.imageBuffer);
		}

		let duration = performance.now() - tStart;
		console.log("duration: " + duration  + "ms");

		postMessage(message, transferables);
		meshIndex++;
	}
	
};