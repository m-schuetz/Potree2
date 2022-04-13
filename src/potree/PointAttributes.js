
const PointAttributeTypes = {
	DOUBLE: {ordinal: 0, name: "double", size: 8},
	FLOAT:  {ordinal: 1, name: "float",  size: 4},
	INT8:   {ordinal: 2, name: "int8",   size: 1},
	UINT8:  {ordinal: 3, name: "uint8",  size: 1},
	INT16:  {ordinal: 4, name: "int16",  size: 2},
	UINT16: {ordinal: 5, name: "uint16", size: 2},
	INT32:  {ordinal: 6, name: "int32",  size: 4},
	UINT32: {ordinal: 7, name: "uint32", size: 4},
	INT64:  {ordinal: 8, name: "int64",  size: 8},
	UINT64: {ordinal: 9, name: "uint64", size: 8}
};

export {PointAttributeTypes};

export class PointAttribute{
	
	constructor(name, type, numElements){
		this.name = name;
		this.type = type;
		this.numElements = numElements;
		this.byteOffset = null;
		this.byteSize = this.numElements * this.type.size;
		this.description = "";
		this.scale = [1.0, 1.0, 1.0];
		this.offset = [0.0, 0.0, 0.0];
		this.range = [Infinity, -Infinity];
	}

};

export class PointAttributes{

	constructor(pointAttributes){
		this.attributes = [];
		this.byteSize = 0;
		this.size = 0;
		this.vectors = [];

		if (pointAttributes != null) {
			for (let i = 0; i < pointAttributes.length; i++) {
				let pointAttribute = pointAttributes[i];
				this.attributes.push(pointAttribute);
				pointAttribute.byteOffset = this.byteSize;
				this.byteSize += pointAttribute.byteSize;
				this.size++;
			}
		}
	}


	add(pointAttribute){
		this.attributes.push(pointAttribute);
		pointAttribute.byteOffset = this.byteSize;
		this.byteSize += pointAttribute.byteSize;
		this.size++;
	};

}