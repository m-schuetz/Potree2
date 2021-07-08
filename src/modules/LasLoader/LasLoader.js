
import {Vector3} from "potree";

export class Header{

	constructor(){
		this.versionMajor = 0;
		this.versionMinor = 0;
		this.headerSize = 0;
		this.offsetToPointData = 0;
		this.format = 0;
		this.recordLength = 0;
		this.numPoints = 0;
		this.scale = new Vector3();
		this.offset = new Vector3();
		this.min = new Vector3();
		this.max = new Vector3();
	}

};

function parseHeader(view){
	let header = new Header();

	header.versionMajor = view.getUint8(24);
	header.versionMinor = view.getUint8(25);
	header.headerSize = view.getUint16(94, true);
	header.offsetToPointData = view.getUint32(96, true);
	header.format = view.getUint8(104);
	header.recordLength = view.getUint16(105, true);
	
	if(header.versionMajor == 1 && header.versionMinor < 4){
		header.numPoints = view.getUint32(107, true);
	}else{
		header.numPoints = Number(view.getBigUint64(247, true));
	}

	header.scale.x = view.getFloat64(131, true);
	header.scale.y = view.getFloat64(139, true);
	header.scale.z = view.getFloat64(147, true);

	header.offset.x = view.getFloat64(155, true);
	header.offset.y = view.getFloat64(163, true);
	header.offset.z = view.getFloat64(171, true);

	header.min.x = view.getFloat64(187, true);
	header.min.y = view.getFloat64(203, true);
	header.min.z = view.getFloat64(219, true);

	header.max.x = view.getFloat64(179, true);
	header.max.y = view.getFloat64(195, true);
	header.max.z = view.getFloat64(211, true);


	return header;
}

async function load(path){

	let response = await fetch(path);
	let buffer = await response.arrayBuffer();
	let view = new DataView(buffer);

	let header = parseHeader(view);

	let {scale, offset} = header;

	let position = new Float64Array(3 * header.numPoints);
	let color = new Uint8Array(4 * header.numPoints);

	for(let i = 0; i < header.numPoints; i++){
		let pointOffset = header.offsetToPointData + i * header.recordLength;

		let X = view.getInt32(pointOffset + 0, true);
		let Y = view.getInt32(pointOffset + 4, true);
		let Z = view.getInt32(pointOffset + 8, true);

		let x = (X * scale.x) + offset.x;
		let y = (Y * scale.y) + offset.y;
		let z = (Z * scale.z) + offset.z;

		position[3 * i + 0] = x;
		position[3 * i + 1] = y;
		position[3 * i + 2] = z;

		let [r, g, b] = [255, 0, 0];

		if(header.format === 3){
			r = view.getUint16(pointOffset + 28, true);
			g = view.getUint16(pointOffset + 30, true);
			b = view.getUint16(pointOffset + 32, true);

			r = r > 255 ? r / 256 : r;
			g = g > 255 ? g / 256 : g;
			b = b > 255 ? b / 256 : b;
		}

		color[4 * i + 0] = r;
		color[4 * i + 1] = g;
		color[4 * i + 2] = b;
		color[4 * i + 3] = 255;
	}

	let positionf32 = new Float32Array(position);

	let buffers = {position, positionf32, color};

	return {header, buffers};
}

export const LasLoader = {
	load
};