
// export const SCALAR = {
// 	name: "scalar",
// 	condition: (attribute) => (attribute.name === "intensity"),
// 	wgsl: `
// 		fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

// 			var offset = node.numPoints * attrib.offset + 2u * pointID;
// 			var value = f32(readU16(offset));

// 			var w = (value - attrib.range_min) / (attrib.range_max - attrib.range_min);

// 			var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
// 			var uv : vec2<f32> = vec2<f32>(w, 0.0);

// 			if(attrib.clamp == CLAMP_ENABLED){
// 				color = textureSampleLevel(gradientTexture, sampler_clamp, uv, 0.0);
// 			}else{
// 				color = textureSampleLevel(gradientTexture, sampler_repeat, uv, 0.0);
// 			}

// 			return color;
// 		}
// 	`,
// };

export const POSITION = {
	name: "position",
	condition: (attribute) => (attribute.name === "position"),
	wgsl: `fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

		var worldPos = (uniforms.world * position);
		var octreeSize = uniforms.octreeMax - uniforms.octreeMin;

		var ux = (worldPos.x - uniforms.octreeMin.x) / octreeSize.x;
		var uy = (worldPos.y - uniforms.octreeMin.y) / octreeSize.y;
		var uz = (worldPos.z - uniforms.octreeMin.z) / octreeSize.z;

		var color = vec4f(ux, uy, uz, 1.0);

		return color;
	}`
};

export const ELEVATION = {
	name: "elevation",
	condition: (attribute) => (attribute.name === "elevation"),
	wgsl: `fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

		var value = (uniforms.world * position).z;
		var w = (value - attrib.range_min) / (attrib.range_max - attrib.range_min);

		var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
		var uv : vec2<f32> = vec2<f32>(w, 0.0);

		if(attrib.clamp == CLAMP_ENABLED){
			color = textureSampleLevel(gradientTexture, sampler_clamp, uv, 0.0);
		}else{
			color = textureSampleLevel(gradientTexture, sampler_repeat, uv, 0.0);
		}

		return color;
	}`
};

export const LAS_INTENSITY_GRADIENT = {
	name: "intensity - gradient",
	condition: (attribute) => (attribute.name === "intensity"),
	wgsl: `
		fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

			var offset = node.numPoints * attrib.offset + 2u * pointID;
			var value = f32(readU16(offset));

			var w = (value - attrib.range_min) / (attrib.range_max - attrib.range_min);

			var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
			var uv : vec2<f32> = vec2<f32>(w, 0.0);

			if(attrib.clamp == CLAMP_ENABLED){
				color = textureSampleLevel(gradientTexture, sampler_clamp, uv, 0.0);
			}else{
				color = textureSampleLevel(gradientTexture, sampler_repeat, uv, 0.0);
			}

			return color;
		}
	`,
};

export const LAS_INTENSITY_DIRECT = {
	name: "intensity - direct",
	condition: (attribute) => (attribute.name === "intensity"),
	wgsl: `
		fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

			var offset = node.numPoints * attrib.offset + 2u * pointID;
			var value = f32(readU16(offset));

			var w = f32(value) / 256.0;
			var color = vec4<f32>(w, w, w, 1.0);

			return color;
		}
	`,
};

export const LAS_CLASSIFICATION = {
	name: "classification",
	condition: (attribute) => (attribute.name === "classification"),
	wgsl: `
		fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

			var offset = node.numPoints * attrib.offset + pointID;

			var value = readU8(offset);

			var color_u32 = colormap[value];

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
	`,
};

export const LAS_RGB = {
	name: "rgba",
	condition: (attribute) => (attribute.name === "rgba"),
	wgsl: `
		fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

			var offset = node.numPoints * attrib.offset + attrib.byteSize * pointID;

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

			if(r > 255.0) { r = r / 256.0; }
			if(g > 255.0) { g = g / 256.0; }
			if(b > 255.0) { b = b / 256.0; }

			var color = vec4<f32>(r, g, b, 255.0) / 255.0;

			return color;
		}
	`,
};

export const LAS_GPS_TIME = {
	name: "gps-time",
	condition: (attribute) => (attribute.name === "gps-time"),
	wgsl: `
		fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {

			var offset = node.numPoints * attrib.offset + 8u * pointID;
			var value = readF64(offset);

			var w = (value - attrib.range_min) / (attrib.range_max - attrib.range_min);

			var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
			var uv : vec2<f32> = vec2<f32>(w, 0.0);

			if(attrib.clamp == CLAMP_ENABLED){
				color = textureSampleLevel(gradientTexture, sampler_clamp, uv, 0.0);
			}else{
				color = textureSampleLevel(gradientTexture, sampler_repeat, uv, 0.0);
			}

			return color;
		}
	`,
};

export const TRIMBLE_NORMAL = {
	name: "normal (trimble 2-15-15)",
	condition: (attribute) => (attribute.description === "Normal vector 2+15+15 bits"),
	wgsl: `
		fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {
			var PI = 3.1415;
			var HML = (2.0 * PI) / 32767.0;
			var VML = PI / 32767.0;
			
			var offset = node.numPoints * attrib.offset + 4u * pointID;
			var value = readU32(offset);

			var mask_15b = (1u << 15u) - 1u;

			var dim = value & 3u;
			var horzAngle = f32((value >>  2u) & mask_15b);
			var vertAngle = f32((value >> 17u) & mask_15b);

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

			return color;
		}
	`,
};

export const TRIMBLE_GROUP = {
	name: "group (trimble)",
	condition: (attribute) => (attribute.name === "Group"),
	wgsl: `
		fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {
			var offset = node.numPoints * attrib.offset + 4u * pointID;
			var value = readU32(offset);

			var w = f32(value) / 1234.0;
			w = f32(value % 10u) / 10.0;
			var uv = vec2<f32>(w, 0.0);

			var color = textureSampleLevel(gradientTexture, sampler_repeat, uv, 0.0);

			return color;
		}
	`,
};

export const TRIMBLE_DISTANCE = {
	name: "distance (trimble)",
	condition: (attribute) => (attribute.name === "Distance"),
	wgsl: `
		fn mapping(pointID : u32, attrib : AttributeDescriptor, node : Node, position : vec4<f32>) -> vec4<f32> {
			var offset = node.numPoints * attrib.offset + 4u * pointID;
			var value = readI32(offset);

			// assuming distance in meters
			var distance = f32(value) / 1000.0;
			var w = distance / 30.0;
			var uv = vec2<f32>(w, 0.0);

			var color = textureSampleLevel(gradientTexture, sampler_clamp, uv, 0.0);

			return color;
		}
	`,
};


export const MAPPINGS = {
	POSITION, 
	ELEVATION,
	LAS_INTENSITY_GRADIENT, 
	LAS_INTENSITY_DIRECT, 
	LAS_CLASSIFICATION, 
	LAS_RGB,
	LAS_GPS_TIME,
	TRIMBLE_NORMAL,
	TRIMBLE_GROUP,
	TRIMBLE_DISTANCE,
};