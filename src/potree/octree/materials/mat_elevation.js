
const wgsl = `

fn map_rgb(vertex : VertexInput, attrib : AttributeDescriptor, node : Node) -> vec4<f32>{

	var offset = node.numPoints * attrib.offset + attrib.byteSize * vertex.vertexID;

	var r = 0.0;
	var g = 0.0;
	var b = 0.0;

	if(attrib.datatype == TYPES_UINT8){
		r = f32(readU8(offset + 0u));
		g = f32(readU8(offset + 1u));
		b = f32(readU8(offset + 2u));
	}else if(attrib.datatype == TYPES_UINT16){
		r = f32(readU16(offset + 0u));
		g = f32(readU16(offset + 2u));
		b = f32(readU16(offset + 4u));
	}

	if(r > 255.0){
		r = r / 256.0;
	}
	if(g > 255.0){
		g = g / 256.0;
	}
	if(b > 255.0){
		b = b / 256.0;
	}

	color = vec4<f32>(r, g, b, 255.0) / 255.0;

	return color;
}

`;

class RgbMaterial{

	constructor(){
		this.attributeIndex = 0;
		this.wgsl = wgsl;
	}

}