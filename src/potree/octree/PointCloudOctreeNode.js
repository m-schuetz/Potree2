
import {Vector3, Box3} from "potree";

export class PointCloudOctreeNode{
	constructor(name){
		this.name = name;
		this.loaded = false;
		this.unfilteredLoaded = false;
		this.parent = null;
		this.children = new Array(8).fill(null);
		this.level = 0;
		// this.isLeaf = true;

		this.boundingBox = new Box3();
	}

	traverse(callback){

		callback(this);

		for(let child of this.children){
			if(child){
				child.traverse(callback);
			}
		}
	}

	getPoint_Voxel(pointID){
		let point = {
			position: new Vector3(0, 0, 0),
			color: new Vector3(0, 0, 0),
			attributes: [],
		};

		if(!this.geometry) return point;

		let geometry = this.geometry;
		let {min, max} = this.boundingBox;
		let view = new DataView(geometry.buffer);

		{ // POSITION
			let X = view.getUint8(3 * pointID + 0);
			let Y = view.getUint8(3 * pointID + 1);
			let Z = view.getUint8(3 * pointID + 2);

			point.position.x = (max.x - min.x) * (X / 128.0) + min.x + this.octree.position.x;
			point.position.y = (max.y - min.y) * (Y / 128.0) + min.y + this.octree.position.y;
			point.position.z = (max.z - min.z) * (Z / 128.0) + min.z + this.octree.position.z;
		}

		{ // COLOR
			let blockSize     = 8;
			let bytesPerBlock = 8;
			let bitsPerSample = 2;
			let numSamples    = 4;
			let blockIndex    = Math.floor(pointID / bytesPerBlock);
			let offset        = geometry.numElements * 3;

			let start_x = view.getUint8(offset + bytesPerBlock * blockIndex + 0);
			let start_y = view.getUint8(offset + bytesPerBlock * blockIndex + 1);
			let start_z = view.getUint8(offset + bytesPerBlock * blockIndex + 2);

			let end_x = view.getUint8(offset + bytesPerBlock * blockIndex + 3);
			let end_y = view.getUint8(offset + bytesPerBlock * blockIndex + 4);
			let end_z = view.getUint8(offset + bytesPerBlock * blockIndex + 5);

			let bits = view.getUint16(offset + bytesPerBlock * blockIndex + 6);

			let sampleIndex = pointID % blockSize;

			let T = (bits >> (bitsPerSample * sampleIndex)) & 3;
			let t = Math.floor((T / (numSamples - 1)));

			let dx = end_x - start_x;
			let dy = end_y - start_y;
			let dz = end_z - start_z;

			let x = Math.floor(dx * t + start_x);
			let y = Math.floor(dy * t + start_y);
			let z = Math.floor(dz * t + start_z);

			point.color.x = x;
			point.color.y = y;
			point.color.z = z;

			point.attributes["rgba"] = [x, y, z];
		}

		return point;
	}

	getPoint_Point(pointID){
		let point = {
			position: new Vector3(0, 0, 0),
			color: new Vector3(0, 0, 0),
			attributes: {},
		};

		if(!this.geometry) return point;

		let geometry = this.geometry;
		let {min, max} = this.boundingBox;
		let view = new DataView(geometry.buffer);

		let readers = {
			"double"  : view.getFloat64.bind(view),
			"float"   : view.getFloat32.bind(view),
			"int8"    : view.getInt8.bind(view),
			"uint8"   : view.getUint8.bind(view),
			"int16"   : view.getInt16.bind(view),
			"uint16"  : view.getUint16.bind(view),
			"int32"   : view.getInt32.bind(view),
			"uint32"  : view.getUint32.bind(view),
			"int64"   : view.getBigInt64.bind(view),
			"uint64"  : view.getBigUint64.bind(view),
		};

		{ // POSITION
			let X = view.getFloat32(12 * pointID + 0, true);
			let Y = view.getFloat32(12 * pointID + 4, true);
			let Z = view.getFloat32(12 * pointID + 8, true);

			point.position.x = X + this.octree.position.x;
			point.position.y = Y + this.octree.position.y;
			point.position.z = Z + this.octree.position.z;
		}

		let attributes = this.octree.loader.attributes;
		let offset = 0;
		for(let i = 0; i < attributes.attributes.length; i++){
			let attribute = attributes.attributes[i];

			let reader = readers[attribute.type.name].bind(view);

			let value = [];
			for(let j = 0; j < attribute.numElements; j++){
				let _value = reader(offset + (pointID * attribute.numElements + j) * attribute.type.size, true);
				value.push(_value);
			}

			point.attributes[attribute.name] = value;

			offset += geometry.numElements * attribute.byteSize;
		}

		return point;
	}

	getPoint(index){

		let point = {
			position: new Vector3(0, 0, 0),
		};

		if(!this.geometry) return point;

		let geometry = this.geometry;
		let {min, max} = this.boundingBox;
		let view = new DataView(geometry.buffer);

		if(geometry.numVoxels > 0){
			// let X = view.getUint8(3 * index + 0);
			// let Y = view.getUint8(3 * index + 1);
			// let Z = view.getUint8(3 * index + 2);

			// point.position.x = (max.x - min.x) * (X / 128.0) + min.x + this.octree.position.x;
			// point.position.y = (max.y - min.y) * (Y / 128.0) + min.y + this.octree.position.y;
			// point.position.z = (max.z - min.z) * (Z / 128.0) + min.z + this.octree.position.z;

			return this.getPoint_Voxel(index);
		}else{
			// let X = view.getFloat32(12 * index + 0, true);
			// let Y = view.getFloat32(12 * index + 4, true);
			// let Z = view.getFloat32(12 * index + 8, true);

			// point.position.x = X + this.octree.position.x;
			// point.position.y = Y + this.octree.position.y;
			// point.position.z = Z + this.octree.position.z;

			return this.getPoint_Point(index);
		}

		// return point;
	}
}