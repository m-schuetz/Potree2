
struct Uniforms {
	world             : mat4x4<f32>,   //   0
	view              : mat4x4<f32>,   //  64
	proj              : mat4x4<f32>,   // 128
	worldView         : mat4x4<f32>,   // 192
	screen_width      : f32,           // 256
	screen_height     : f32,           // 260
	hqs_flag          : u32,           // 264
	colorMode         : u32,           // 268
	selectedAttribute : u32,           // 272
};

struct Node {
	numPoints   : u32,
	counter     : u32,
	min_x       : f32,
	min_y       : f32,
	min_z       : f32,
	max_x       : f32,
	max_y       : f32,
	max_z       : f32,
};

struct AttributeDescriptor{
	offset      : u32,
	numElements : u32,
	valuetype   : u32,
	range_min   : f32,
	range_max   : f32,
	clamp       : u32,
	byteSize    : u32,
	datatype    : u32,
};

struct Nodes{ values : array<Node> };
struct AttributeDescriptors{ values : array<AttributeDescriptor> };
struct U32s { values : array<u32> };

let COLORMODE_UNDEFINED     = 0u;
let COLORMODE_RGBA          = 1u;
let COLORMODE_SCALAR        = 2u;
let COLORMODE_ELEVATION     = 3u;
let COLORMODE_LISTING       = 4u;

let TYPES_DOUBLE          =  0u;
let TYPES_FLOAT           =  1u;
let TYPES_INT8            =  2u;
let TYPES_UINT8           =  3u;
let TYPES_INT16           =  4u;
let TYPES_UINT16          =  5u;
let TYPES_INT32           =  6u;
let TYPES_UINT32          =  7u;
let TYPES_INT64           =  8u;
let TYPES_UINT64          =  9u;
let TYPES_RGBA            = 50u;
let TYPES_ELEVATION       = 51u;

let CLAMP_DISABLED        =  0u;
let CLAMP_ENABLED         =  1u;

let RED      = vec4<f32>(1.0, 0.0, 0.0, 1.0);
let GREEN    = vec4<f32>(0.0, 1.0, 0.0, 1.0);
let BLUE     = vec4<f32>(0.0, 0.0, 1.0, 1.0);
let OUTSIDE  = vec4<f32>(10.0, 10.0, 10.0, 1.0);

@binding(0) @group(0) var<uniform> uniforms           : Uniforms;
@binding(1) @group(0) var<storage, read> attributes   : AttributeDescriptors;
@binding(2) @group(0) var<storage, read> colormap     : U32s;

@binding(0) @group(1) var sampler_repeat              : sampler;
@binding(1) @group(1) var sampler_clamp               : sampler;
@binding(2) @group(1) var gradientTexture             : texture_2d<f32>;

@binding(0) @group(2) var<storage, read> buffer       : U32s;
@binding(0) @group(3) var<storage, read> nodes        : Nodes;


fn readU8(offset : u32) -> u32{
	var ipos : u32 = offset / 4u;
	var val_u32 : u32 = buffer.values[ipos];

	// var shift : u32 = 8u * (3u - (offset % 4u));
	var shift : u32 = 8u * (offset % 4u);
	var val_u8 : u32 = (val_u32 >> shift) & 0xFFu;

	return val_u8;
}

fn readU16(offset : u32) -> u32{
	
	var first = readU8(offset + 0u);
	var second = readU8(offset + 1u);

	var value = first | (second << 8u);

	return value;
}

fn readI16(offset : u32) -> i32{
	
	var first = readU8(offset + 0u);
	var second = readU8(offset + 1u);

	var sign = second >> 7u;
	second = second & 127u;

	var value = -2;

	if(sign == 0u){
		value = 0;
	}else{
		value = -1;
	}
	var mask = 0xFFFF << 16u;
	value = value & mask;
	value = value | i32(first) << 0u;
	value = value | i32(second) << 8u;

	return value;
}

fn readU32(offset : u32) -> u32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	return value;
}

fn readF32(offset : u32) -> f32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value_u32 = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	var value_f32 = bitcast<f32>(value_u32);

	return value_f32;
}

fn readF64(offset : u32) -> f32{
	
	//var offset = node.numPoints * attribute.offset + 8u * vertex.vertexID;

	var b0 = readU8(offset + 0u);
	var b1 = readU8(offset + 1u);
	var b2 = readU8(offset + 2u);
	var b3 = readU8(offset + 3u);
	var b4 = readU8(offset + 4u);
	var b5 = readU8(offset + 5u);
	var b6 = readU8(offset + 6u);
	var b7 = readU8(offset + 7u);

	var exponent_f64_bin = (b7 << 4u) | (b6 >> 4u);
	var exponent_f64 = exponent_f64_bin - 1023u;

	var exponent_f32_bin = exponent_f64 + 127u;
	var mantissa_f32 = (b6 & 0x0Fu) << 19u
		| b5 << 11u
		| b4 << 3u;
	var sign = (b7 >> 7u) & 1u;
	var value_u32 = sign << 31u
		| exponent_f32_bin << 23u
		| mantissa_f32;

	var value_f32 = bitcast<f32>(value_u32);

	return value_f32;
}

struct VertexInput {
	@builtin(instance_index) instanceID : u32,
	@builtin(vertex_index) vertexID : u32,
};

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec4<f32>,
	@location(1) @interpolate(flat) point_id : u32,
};

fn colorFromListing(vertex : VertexInput, attribute : AttributeDescriptor, node : Node) -> vec4<f32> {
	var offset = node.numPoints * attribute.offset + 1u * vertex.vertexID;

	var value = readU8(offset);

	var color_u32 = colormap.values[value];

	var r = (color_u32 >>  0u) & 0xFFu;
	var g = (color_u32 >>  8u) & 0xFFu;
	var b = (color_u32 >> 16u) & 0xFFu;

	var color = vec4<f32>(
		f32(r) / 255.0,
		f32(g) / 255.0,
		f32(b) / 255.0,
		1.0
	);

	return color;
}

fn map_normal_trimble_2_15_15(vertex : VertexInput, attribute : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

	var PI = 3.1415;
	var HML = (2.0 * PI) / 32767.0;
	var VML = PI / 32767.0;
	
	var offset = node.numPoints * attribute.offset + 4u * vertex.vertexID;
	var value = readU32(offset);

	var mask_15b = (1u << 15u) - 1u;
	var dim = value >> 30u;
	var vertAngle = f32((value >>  0u) & mask_15b);
	var horzAngle = f32((value >> 15u) & mask_15b);

	var ang = (VML * vertAngle) - 0.5 * PI;
	var zvl = sin(ang);
	var xml = sqrt( 1.0 - (zvl * zvl));

	var normal : vec3<f32>;
	normal.x = xml * cos(HML * horzAngle);
	normal.y = xml * sin(HML * horzAngle);
	normal.z = zvl;

	var color = vec4<f32>(normal, 1.0);

	color = vec4<f32>(
		1.0 * normal.x, 
		1.0 * normal.y, 
		1.0 * normal.z,
		1.0,
	);

	// if(dim == 0u){
	// 	color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
	// }else if(dim == 1u){
	// 	color = vec4<f32>(0.0, 1.0, 0.0, 1.0);
	// }else if(dim == 2u){
	// 	color = vec4<f32>(0.0, 0.0, 1.0, 1.0);
	// }else if(dim == 3u){
	// 	color = vec4<f32>(1.0, 0.0, 1.0, 1.0);
	// }

	return color;
}

fn scalarToColor(vertex : VertexInput, attribute : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

	var value : f32 = 0.0;

	if(attribute.valuetype == TYPES_UINT8){
		var offset = node.numPoints * attribute.offset + 1u * vertex.vertexID;
		value = f32(readU8(offset));
	}else if(attribute.valuetype == TYPES_DOUBLE){
		var offset = node.numPoints * attribute.offset + 8u * vertex.vertexID;

		value = readF64(offset);
	}else if(attribute.valuetype == TYPES_ELEVATION){
		value = (uniforms.world * position).z;
	}else if(attribute.valuetype == TYPES_UINT16){
		var offset = node.numPoints * attribute.offset + 2u * vertex.vertexID;
		value = f32(readU16(offset));
	}else if(attribute.valuetype == TYPES_INT16){
		var offset = node.numPoints * attribute.offset + 2u * vertex.vertexID;
		value = f32(readI16(offset));
	}else if(attribute.valuetype == TYPES_UINT32){
		var offset = node.numPoints * attribute.offset + 4u * vertex.vertexID;
		value = f32(readU32(offset));
	}

	var w = (value - attribute.range_min) / (attribute.range_max - attribute.range_min);

	var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
	var uv : vec2<f32> = vec2<f32>(w, 0.0);

	

	if(attribute.clamp == CLAMP_ENABLED){
		color = textureSampleLevel(gradientTexture, sampler_clamp, uv, 0.0);
	}else{
		color = textureSampleLevel(gradientTexture, sampler_repeat, uv, 0.0);
	}

	// {
	// 	var w = (value - 20000.0) / 40000.0;
	// 	// w = value / 20000.0;

	// 	if(value < 0.0){
	// 		w = 0.0;
	// 	}else{
	// 		w = 0.7;
	// 	}

	// 	var uv : vec2<f32> = vec2<f32>(w, 0.0);
	// 	color = textureSampleLevel(gradientTexture, sampler_clamp, uv, 0.0);
	// }

	// {
	// 	color.a = 1.0;
	// 	// var aIntensity = attributes.values[1];
	// 	var aPointSource = attributes.values[8];
	// 	var offset = node.numPoints * aPointSource.offset + 2u * vertex.vertexID;
	// 	var sourceID = readU16(offset);

	// 	// if(w > 0.5)
	// 	if(sourceID < 120u){
	// 		color.a = 0.0;
	// 	}
	// }

	// color.r = 1.0;

	return color;
}

fn vectorToColor(vertex : VertexInput, attribute : AttributeDescriptor, node : Node) -> vec4<f32> {

	var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

	if(attribute.valuetype == TYPES_RGBA){
		var offset = node.numPoints * attribute.offset + attribute.byteSize * vertex.vertexID;
		// var offset = 29u * node.numPoints + 4u * vertex.vertexID;

		var r = 0.0;
		var g = 0.0;
		var b = 0.0;

		if(attribute.datatype == TYPES_UINT8){
			r = f32(readU8(offset + 0u));
			g = f32(readU8(offset + 1u));
			b = f32(readU8(offset + 2u));
		}else if(attribute.datatype == TYPES_UINT16){
			r = f32(readU16(offset + 0u));
			g = f32(readU16(offset + 2u));
			b = f32(readU16(offset + 4u));
		}

		// {
		// 	r = f32(readU8(offset + 0u));
		// 	g = f32(readU8(offset + 1u));
		// 	b = f32(readU8(offset + 2u));
		// }

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
	}if(attribute.valuetype == TYPES_DOUBLE){
		var offset = node.numPoints * attribute.offset + attribute.byteSize * vertex.vertexID;

		var x = readF64(offset +  0u);
		var y = readF64(offset +  8u);
		var z = readF64(offset + 16u);

		color = vec4<f32>(x, y, z, 1.0);
	}

	return color;
}

fn doIgnores(){

	_ = uniforms;
	_ = &attributes;
	_ = sampler_repeat;
	_ = sampler_clamp;
	_ = gradientTexture;
	_ = &buffer;
	_ = &nodes;

}

@stage(vertex)
fn main_vertex(vertex : VertexInput) -> VertexOutput {

	doIgnores();

	var node = nodes.values[vertex.instanceID];

	var output : VertexOutput;

	var position : vec4<f32>;
	var viewPos : vec4<f32>;
	{

		{ // 3xFLOAT
			var offset = 12u * vertex.vertexID;
			
			position = vec4<f32>(
				readF32(offset + 0u),
				readF32(offset + 4u),
				readF32(offset + 8u),
				1.0,
			);
			
			viewPos = uniforms.worldView * position;
			output.position = uniforms.proj * viewPos;	
		}
		

		// { // compressed coordinates. 1xUINT32
		// 	// var offset = 4u * vertex.vertexID;
		// 	var encoded = readU32( 4u * vertex.vertexID);

		// 	var cubeSize = node.max_x - node.min_x;
		// 	var X = (encoded >> 20u) & 0x3ffu;
		// 	var Y = (encoded >> 10u) & 0x3ffu;
		// 	var Z = (encoded >>  0u) & 0x3ffu;

		// 	var x = (f32(X) / 1024.0) * cubeSize + node.min_x;
		// 	var y = (f32(Y) / 1024.0) * cubeSize + node.min_y;
		// 	var z = (f32(Z) / 1024.0) * cubeSize + node.min_z;
		// 	position = vec4<f32>(x, y, z, 1.0);

		// 	viewPos = uniforms.worldView * position;
		// 	output.position = uniforms.proj * viewPos;	
		// }

		

		// if(vertex.instanceID != 0u){
		// 	output.position = OUTSIDE;
		// }
	}

	// in the HQS depth pass, shift points 1% further away from camera
	if(uniforms.hqs_flag > 0u){
		output.position = output.position / output.position.w;

		viewPos.z = viewPos.z * 1.01;
		
		var shifted : vec4<f32> = uniforms.proj * viewPos;
		output.position.z = shifted.z / shifted.w;
	}

	{ // COLORIZE BY ATTRIBUTE DESCRIPTORS
		var attribute = attributes.values[uniforms.selectedAttribute];
		var value : f32 = 0.0;

		var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

		if(uniforms.colorMode == COLORMODE_LISTING){
			color = colorFromListing(vertex, attribute, node);
		}else if(attribute.numElements == 1u){
			color = scalarToColor(vertex, attribute, node, position);
			// color = map_normal_trimble_2_15_15(vertex, attribute, node, position);
		}else{
			color = vectorToColor(vertex, attribute, node);
		}


		output.color = color;
	}

	if(output.color.a == 0.0){
		output.position.w = 0.0;
	}

	//output.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);

	// bit pattern
	// 12 node counter : 20 point id
	
	// output.point_id = vertex.vertexID;
	// output.point_id = ((node.counter & 0xFFFu) << 20u) | vertex.vertexID;
	output.point_id = node.counter + vertex.vertexID;

	return output;
}

struct FragmentInput {
	@location(0) color : vec4<f32>,
	@location(1) @interpolate(flat) point_id : u32,
};

struct FragmentOutput {
	@location(0) color : vec4<f32>,
	@location(1) point_id : u32,
};

@stage(fragment)
fn main_fragment(fragment : FragmentInput) -> FragmentOutput {

	doIgnores();

	var output : FragmentOutput;
	output.color = fragment.color;
	output.point_id = fragment.point_id;

	return output;
}