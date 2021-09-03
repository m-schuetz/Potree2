
// import {Vector3, Box3} from "potree";
import {Vector3, Box3} from "../math/math.js";

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

	// console.log(json);

	let binaryChunkOffset = 12 + 8 + chunkSize + 8;
	let binaryChunkSize = view.getUint32(20 + chunkSize, true);
	
	let bufferViews = [];
	for(let glbBufferView of json.bufferViews){

		let offset = glbBufferView.byteOffset;
		let size = glbBufferView.byteLength;

		let data = buffer.slice(binaryChunkOffset + offset, binaryChunkOffset + offset + size);

		bufferViews.push({buffer: data});
	}

	// let geometry = new Geometry();
	let geometry = {
		buffers: [],
		indices: null,
		numElement: 0,
	};

	let glbMesh = json.meshes[0];
	let glbPrimitive = glbMesh.primitives[0];
	for(let attributeName of Object.keys(glbPrimitive.attributes)){
		let accessorRef = glbPrimitive.attributes[attributeName];
		let accessor = json.accessors[accessorRef];
		let bufferView = bufferViews[accessor.bufferView];

		let mappedName = {
			"COLOR_0": "color",
			"NORMAL": "normal",
			"POSITION": "position",
			"TEXCOORD_0": "uv",
		}[attributeName] ?? attributeName;

		let geomBuffer = {name: mappedName, buffer: new Uint8Array(bufferView.buffer)};
		geometry.buffers.push(geomBuffer);

		if(glbPrimitive.indices){
			let accessor = json.accessors[glbPrimitive.indices];
			let bufferView = bufferViews[accessor.bufferView];

			geometry.indices = new Uint32Array(bufferView.buffer);
		}
	}

	if(true)
	{ // DEBUG: de-indexing

		let indices = geometry.indices;
		let numTriangles = indices.length / 3;
		let inPositions = new Float32Array(geometry.buffers.find(b => b.name === "position").buffer.buffer);
		let inColors = new Uint32Array(geometry.buffers.find(b => b.name === "color").buffer.buffer);

		let hasUVs = geometry.buffers.find(b => b.name === "uv") ? true : false;
		let hasNormals = geometry.buffers.find(b => b.name === "normal") ? true : false;

		let inUVs = hasUVs ? new Uint32Array(geometry.buffers.find(b => b.name === "uv").buffer.buffer) : null;
		let inNormals = hasNormals ? new Uint32Array(geometry.buffers.find(b => b.name === "normal").buffer.buffer) : null;
		let outPositions = new Float32Array(9 * numTriangles);
		let outColors = new Uint32Array(3 * numTriangles);
		let outIndices = new Uint32Array(indices.length);
		let outUVs = new Float32Array(2 * 3 * numTriangles);
		let outNormals = new Float32Array(3 * 3 * numTriangles);

		for(let i = 0; i < indices.length; i++){

			let inIndex = indices[i];

			outPositions[3 * i + 0] = inPositions[3 * inIndex + 0];
			outPositions[3 * i + 1] = inPositions[3 * inIndex + 1];
			outPositions[3 * i + 2] = inPositions[3 * inIndex + 2];

			outColors[i] = inColors[inIndex];

			if(hasUVs){
				outUVs[2 * i + 0] = inUVs[2 * inIndex + 0];
				outUVs[2 * i + 1] = inUVs[2 * inIndex + 1];
			}

			if(hasNormals){
				outNormals[3 * i + 0] = inNormals[3 * inIndex + 0];
				outNormals[3 * i + 1] = inNormals[3 * inIndex + 1];
				outNormals[3 * i + 2] = inNormals[3 * inIndex + 2];
			}

			outIndices[i] = i;
		}

		// geometry.buffers.find(b => b.name === "position").buffer = new Uint8Array(outPositions.buffer);
		geometry.buffers.find(b => b.name === "position").buffer = new Float32Array(outPositions.buffer);
		geometry.buffers.find(b => b.name === "color").buffer = new Uint32Array(outColors.buffer);

		if(hasUVs){
			geometry.buffers.find(b => b.name === "uv").buffer = new Uint8Array(outUVs.buffer);
		}

		if(hasNormals){
			geometry.buffers.find(b => b.name === "normal").buffer = new Uint8Array(outNormals.buffer);
		}

		geometry.indices = outIndices;
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
		let image = json.images[0];
		let buffer = bufferViews[image.bufferView].buffer;
		let mimeType = image.mimeType;

		var u8 = new Uint8Array(buffer);
		var blob = new Blob([u8], {type: mimeType});

		imageBitmap = await createImageBitmap(blob);
		imageBuffer = buffer;
	}

	let message = {
		geometry: geometry,
		imageBitmap: imageBitmap,
		imageBuffer: imageBuffer,
	};

	let transferables = [
		...geometry.buffers.map(b => b.buffer.buffer),
		geometry.indices.buffer,
	];

	if(imageBitmap){
		transferables.push(imageBitmap);
		transferables.push(imageBuffer);
	}

	let duration = performance.now() - tStart;
	console.log("duration: " + duration  + "ms");

	postMessage(message, transferables);
};