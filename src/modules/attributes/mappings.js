
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

export const MAPPINGS = {
	LAS_INTENSITY_GRADIENT, 
	LAS_INTENSITY_DIRECT, 
	LAS_CLASSIFICATION, 
	LAS_RGB,
};