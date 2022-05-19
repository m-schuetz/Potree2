
struct Uniforms {
	world             : mat4x4<f32>,   //   0
	view              : mat4x4<f32>,   //  64
	proj              : mat4x4<f32>,   // 128
	worldView         : mat4x4<f32>,   // 192
	screen_width      : f32,           // 256
	screen_height     : f32,           // 260
	hqs_flag          : u32,           // 264
	selectedAttribute : u32,           // 268
	time              : f32,           // 272
	pointSize         : f32,           // 276
	splatType         : u32,           // 280
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
	mapping     : u32,
};

struct Nodes{ values : array<Node> };
struct AttributeDescriptors{ values : array<AttributeDescriptor> };
struct U32s { values : array<u32> };

let MAPPING_UNDEFINED     =  0u;
let MAPPING_RGBA          =  1u;
let MAPPING_SCALAR        =  2u;
let MAPPING_ELEVATION     =  3u;
let MAPPING_LISTING       =  4u;
let MAPPING_VECTOR        =  5u;
let MAPPING_CUSTOM        =  6u;
<<TEMPLATE_MAPPING_ENUM>>

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


fn readI32(offset : u32) -> i32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	return i32(value);
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
	@location(2) @interpolate(flat) point_position : vec4<f32>,
};

fn map_listing(vertex : VertexInput, pointID : u32, attribute : AttributeDescriptor, node : Node) -> vec4<f32> {
	var offset = node.numPoints * attribute.offset + 1u * pointID;

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

// fn map_normal_terrascan_2_15_15(vertex : VertexInput, attribute : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {
// 	var PI = 3.1415;
// 	var HML = (2.0 * PI) / 32767.0;
// 	var VML = PI / 32767.0;
	
// 	var offset = node.numPoints * attribute.offset + 4u * vertex.vertexID;
// 	var value = readU32(offset);

// 	var mask_15b = (1u << 15u) - 1u;

// 	var dim = value & 3u;
// 	var horzAngle = f32((value >>  2u) & mask_15b);
// 	var vertAngle = f32((value >> 17u) & mask_15b);

// 	var ang = (VML * vertAngle) - 0.5 * PI;
// 	var zvl = sin(ang);
// 	var xml = sqrt( 1.0 - (zvl * zvl));

// 	var normal : vec3<f32>;
// 	normal.x = xml * cos(HML * horzAngle);
// 	normal.y = xml * sin(HML * horzAngle);
// 	normal.z = zvl;

// 	var color = vec4<f32>(normal, 1.0);

// 	color = vec4<f32>(
// 		1.0 * normal.x, 
// 		1.0 * normal.y, 
// 		1.0 * normal.z,
// 		1.0,
// 	);

// 	return color;
// }

// fn map_terrascan_group(vertex : VertexInput, attribute : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {
// 	var offset = node.numPoints * attribute.offset + 4u * vertex.vertexID;
// 	var value = readU32(offset);

// 	var w = f32(value) / 1234.0;
// 	w = f32(value % 10u) / 10.0;
// 	var uv = vec2<f32>(w, 0.0);

// 	var color = textureSampleLevel(gradientTexture, sampler_repeat, uv, 0.0);

// 	return color;
// }

// fn map_terrascan_distance(vertex : VertexInput, attribute : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {
// 	var offset = node.numPoints * attribute.offset + 4u * vertex.vertexID;
// 	var value = readI32(offset);

// 	// assuming distance in meters
// 	var distance = f32(value) / 1000.0;
// 	var w = distance / 5.0;
// 	var uv = vec2<f32>(w, 0.0);

// 	var color = textureSampleLevel(gradientTexture, sampler_repeat, uv, 0.0);

// 	return color;
// }


fn map_scalar(vertex : VertexInput, pointID : u32, attribute : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

	var value : f32 = 0.0;

	if(attribute.valuetype == TYPES_UINT8){
		var offset = node.numPoints * attribute.offset + 1u * pointID;
		value = f32(readU8(offset));
	}else if(attribute.valuetype == TYPES_DOUBLE){
		var offset = node.numPoints * attribute.offset + 8u * pointID;

		value = readF64(offset);
	}else if(attribute.valuetype == TYPES_ELEVATION){
		value = (uniforms.world * position).z;
	}else if(attribute.valuetype == TYPES_UINT16){
		var offset = node.numPoints * attribute.offset + 2u * pointID;
		value = f32(readU16(offset));
	}else if(attribute.valuetype == TYPES_INT16){
		var offset = node.numPoints * attribute.offset + 2u * pointID;
		value = f32(readI16(offset));
	}else if(attribute.valuetype == TYPES_UINT32){
		var offset = node.numPoints * attribute.offset + 4u * pointID;
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

	return color;
}

fn map_vector(vertex : VertexInput, pointID : u32, attribute : AttributeDescriptor, node : Node) -> vec4<f32> {

	var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

	if(attribute.valuetype == TYPES_RGBA){
		var offset = node.numPoints * attribute.offset + attribute.byteSize * pointID;

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

		if(r > 255.0) { r = r / 256.0; }
		if(g > 255.0) { g = g / 256.0; }
		if(b > 255.0) { b = b / 256.0; }

		color = vec4<f32>(r, g, b, 255.0) / 255.0;
	}if(attribute.valuetype == TYPES_DOUBLE){
		var offset = node.numPoints * attribute.offset + attribute.byteSize * pointID;

		var x = readF64(offset +  0u);
		var y = readF64(offset +  8u);
		var z = readF64(offset + 16u);

		color = vec4<f32>(x, y, z, 1.0);
	}

	return color;
}

fn map_elevation(vertex : VertexInput, pointID : u32, attribute : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

	var value = (uniforms.world * position).z;
	var w = (value - attribute.range_min) / (attribute.range_max - attribute.range_min);

	var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
	var uv : vec2<f32> = vec2<f32>(w, 0.0);

	if(attribute.clamp == CLAMP_ENABLED){
		color = textureSampleLevel(gradientTexture, sampler_clamp, uv, 0.0);
	}else{
		color = textureSampleLevel(gradientTexture, sampler_repeat, uv, 0.0);
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
	var point_position : vec4<f32>;
	var viewPos : vec4<f32>;

	var pointID = 0u;
	if(uniforms.splatType == 0u){
		pointID = vertex.vertexID;
	}else if(uniforms.splatType == 1u){
		pointID = vertex.vertexID / 6u;
	}

	{

		{ // 3xFLOAT
			var offset = 12u * pointID;
			
			position = vec4<f32>(
				readF32(offset + 0u),
				readF32(offset + 4u),
				readF32(offset + 8u),
				1.0,
			);
			
			viewPos = uniforms.worldView * position;
			output.position = uniforms.proj * viewPos;

			{ // point_position

				var ndcPos = output.position.xy / output.position.w;
				point_position = vec4<f32>(
					(ndcPos.x * 0.5 + 0.5) * uniforms.screen_width,
					(ndcPos.y * 0.5 + 0.5) * uniforms.screen_height,
					0.0, 1.0
				);

			}

			if(uniforms.splatType == 1u){
				var localIndex = vertex.vertexID % 6u;

				var pixelSize = uniforms.pointSize;

				var transX = (pixelSize / uniforms.screen_width) * output.position.w;
				var transY = (pixelSize / uniforms.screen_height) * output.position.w;

				if(localIndex == 0u){
					output.position.x = output.position.x - transX;
					output.position.y = output.position.y - transY;
				}else if(localIndex == 1u){
					output.position.x = output.position.x + transX;
					output.position.y = output.position.y - transY;
				}else if(localIndex == 2u){
					output.position.x = output.position.x + transX;
					output.position.y = output.position.y + transY;
				}else if(localIndex == 3u){
					output.position.x = output.position.x - transX;
					output.position.y = output.position.y - transY;
				}else if(localIndex == 4u){
					output.position.x = output.position.x + transX;
					output.position.y = output.position.y + transY;
				}else if(localIndex == 5u){
					output.position.x = output.position.x - transX;
					output.position.y = output.position.y + transY;
				}
			}

		}

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

		if(attribute.mapping == MAPPING_LISTING){
			color = map_listing(vertex, pointID, attribute, node);
		}else if(attribute.mapping == MAPPING_SCALAR){
			color = map_scalar(vertex, pointID, attribute, node, position);
		}else if(attribute.mapping == MAPPING_ELEVATION){
			color = map_elevation(vertex, pointID, attribute, node, position);
		}else if(attribute.mapping == MAPPING_RGBA){
			color = map_vector(vertex, pointID, attribute, node);
		}else if(attribute.mapping == MAPPING_VECTOR){
			color = map_vector(vertex, pointID, attribute, node);
		}
		<<TEMPLATE_MAPPING_SELECTION>>
		
		// color = map_normal_terrascan_2_15_15(vertex, attribute, node, position);
		// color = map_terrascan_group(vertex, attribute, node, position);
		// color = map_terrascan_distance(vertex, attribute, node, position);

		output.color = color;
	}

	if(output.color.a == 0.0){
		output.position.w = 0.0;
	}

	output.point_id = node.counter + pointID;
	output.point_position = point_position;
 
	return output;
}

struct FragmentInput {
	@location(0) color : vec4<f32>,
	@location(1) @interpolate(flat) point_id : u32,
	@location(2) @interpolate(flat) point_position : vec4<f32>,
	@builtin(position) frag_position : vec4<f32>,
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

	var uv = vec2<f32>(
		fragment.frag_position.x - fragment.point_position.x,
		(uniforms.screen_height - fragment.frag_position.y) - fragment.point_position.y
	);

	if(uniforms.splatType == 0u){
		output.color = vec4<f32>(fragment.color.xyz, 1.0);
	}else if(uniforms.splatType == 1u){
		var d = length(uv / (uniforms.pointSize * 0.5));
		var weight = pow(1.0 - d * d, 8.0);
		weight = clamp(weight, 0.001, 10.0);

		if(d > 1.0){
			weight = 0.0;
			discard;
		}

		var weighted = fragment.color.xyz * weight;

		output.color = vec4<f32>(weighted, weight);
	}

	

	return output;
}

<<TEMPLATE_MAPPING_FUNCTIONS>>