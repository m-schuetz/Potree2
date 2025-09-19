
struct Uniforms {	
	worldView        : mat4x4f,
	world            : mat4x4f,
	view             : mat4x4f,
	proj             : mat4x4f,
	screen_width     : f32,
	screen_height    : f32,
	size             : f32,
	elementCounter   : u32,
	hoveredIndex     : i32,
	numSplats        : u32,
};

@group(0) @binding(0) var<uniform> uniforms    : Uniforms;
@group(0) @binding(1) var<storage> a_ordering  : array<u32>;

@group(0) @binding(2) var<storage> a_positions : array<f32>;
@group(0) @binding(3) var<storage> a_color     : array<vec4f>;
@group(0) @binding(4) var<storage> a_rotation  : array<vec4f>;
@group(0) @binding(5) var<storage> a_scale     : array<f32>;


struct VertexIn{
	@builtin(vertex_index) vertex_index : u32,
	@builtin(instance_index) instance_index : u32,
};

struct VertexOut{
	@builtin(position) position : vec4<f32>,
	@location(1) @interpolate(linear) color : vec4<f32>,
	@location(2) @interpolate(perspective) uv : vec2f,
};

struct FragmentIn{
	@location(1) @interpolate(linear) color : vec4<f32>,
	@location(2) @interpolate(perspective) uv : vec2f,
};

struct FragmentOut{
	@location(0) color : vec4<f32>
};

// Adapted from glm mat3_cast: https://github.com/g-truc/glm/blob/2d4c4b4dd31fde06cfffad7915c2b3006402322f/glm/gtc/quaternion.inl#L47
// Licensed under MIT: https://github.com/g-truc/glm/blob/master/copying.txt
fn toMat3(q : vec4f) -> mat3x3f{

	var qxx = (q.x * q.x);
	var qyy = (q.y * q.y);
	var qzz = (q.z * q.z);
	var qxz = (q.x * q.z);
	var qxy = (q.x * q.y);
	var qyz = (q.y * q.z);
	var qwx = (q.w * q.x);
	var qwy = (q.w * q.y);
	var qwz = (q.w * q.z);

	var mat = mat3x3f(
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f
	);

	mat[0][0] = 1.0f - 2.0f * (qyy +  qzz);
	mat[0][1] = 2.0f * (qxy + qwz);
	mat[0][2] = 2.0f * (qxz - qwy);

	mat[1][0] = 2.0f * (qxy - qwz);
	mat[1][1] = 1.0f - 2.0f * (qxx +  qzz);
	mat[1][2] = 2.0f * (qyz + qwx);

	mat[2][0] = 2.0f * (qxz + qwy);
	mat[2][1] = 2.0f * (qyz - qwx);
	mat[2][2] = 1.0f - 2.0f * (qxx +  qyy);

	return mat;
};


// Much of the splat math originates from https://github.com/mkkellogg/GaussianSplats3D (MIT License)
@vertex
fn main_vertex(vertex : VertexIn) -> VertexOut {


	var localVertexIndex = vertex.vertex_index % 6u;

	var idx = vertex.vertex_index / 6u;
	var splatIndex = a_ordering[idx];

	var splatPos =  vec4f(
		a_positions[3u * splatIndex + 0], 
		a_positions[3u * splatIndex + 1], 
		a_positions[3u * splatIndex + 2], 
		1.0f);
	var worldPos = uniforms.world * splatPos;
	var viewPos = uniforms.view * worldPos;
	var ndc = uniforms.proj * viewPos;

	ndc.x = ndc.x / ndc.w;
	ndc.y = ndc.y / ndc.w;
	ndc.z = ndc.z / ndc.w;

	var q = a_rotation[splatIndex];
	var rotation = toMat3(q);

	var vscale = vec3f(
		a_scale[3u * splatIndex + 0],
		a_scale[3u * splatIndex + 1],
		a_scale[3u * splatIndex + 2]
	);
	var scale = mat3x3f(
		vscale.x, 0.0f, 0.0f, 
		0.0f, vscale.y, 0.0f, 
		0.0f, 0.0f, vscale.z
	);

	var cov3D = rotation * scale * scale * transpose(rotation);


	var focal = vec2f(
		uniforms.proj[0][0] * uniforms.screen_width * 0.5f,
		uniforms.proj[1][1] * uniforms.screen_height * 0.5f
	);
	var s = 1.0f / (viewPos.z * viewPos.z);
	var J = mat3x3f(
		focal.x / viewPos.z   , 0.0f                  , -(focal.x * viewPos.x) * s,
		0.0f                  , focal.y / viewPos.z   , -(focal.y * viewPos.y) * s,
		0.0f                  , 0.0f                  , 0.0f
	);

	var W = transpose(mat3x3f(uniforms.worldView[0].xyz, uniforms.worldView[1].xyz, uniforms.worldView[2].xyz));
	var T = W * J;

	var cov2Dm = transpose(T) * cov3D * T;
	cov2Dm[0][0] += 0.3f;
	cov2Dm[1][1] += 0.3f;

	var cov2Dv = vec3f(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

	var a = cov2Dv.x;
	var d = cov2Dv.z;
	var b = cov2Dv.y;
	var D = a * d - b * b;
	var trace = a + d;
	var traceOver2 = 0.5f * trace;
	var term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
	var eigenValue1 = traceOver2 + term2;
	var eigenValue2 = traceOver2 - term2;

	eigenValue1 = max(eigenValue1, 0.001f);
	eigenValue2 = max(eigenValue2, 0.001f);

	var eigenVector1 = normalize(vec2f(b, eigenValue1 - a));
	var eigenVector2 = vec2f(eigenVector1.y, -eigenVector1.x);

	// float splatScale = args.uniforms.splatSize;
	var splatScale = 1.0f;
	var sqrt8 = sqrt(8.0f);

	// if(length(vscale) < 0.4f){
	// 	eigenValue1 = 0.0f;
	// }

	// if(length(vscale) < 0.44f || length(vscale) > 0.45654f){
	// 	eigenValue1 = 0.0f;
	// }

	// // if(splatIndex < 28111){
	// if(splatIndex == 28111){
	// 	eigenValue1 = 0.0f;
	// }

	var MAX_SCREENSPACE_SPLATSIZE = 300.0f;
	var basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), MAX_SCREENSPACE_SPLATSIZE);
	var basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), MAX_SCREENSPACE_SPLATSIZE);

	var vout = VertexOut();

	{
		var depth = (uniforms.proj * uniforms.worldView * worldPos).w;
		var aspect = uniforms.screen_width / uniforms.screen_height;
		var dx = aspect * depth / uniforms.screen_width;
		var dy = depth / uniforms.screen_height;

		var A = vec4f(basisVector1.x * dx, basisVector1.y * dy, 0.0f, 0.0f);
		var B = vec4f(basisVector2.x * dx, basisVector2.y * dy, 0.0f, 0.0f);

		// POS
		if(localVertexIndex == 0u) { viewPos = viewPos - A - B;};
		if(localVertexIndex == 1u) { viewPos = viewPos + A - B;};
		if(localVertexIndex == 2u) { viewPos = viewPos + A + B;};
		
		if(localVertexIndex == 3u) { viewPos = viewPos - A - B;};
		if(localVertexIndex == 4u) { viewPos = viewPos + A + B;};
		if(localVertexIndex == 5u) { viewPos = viewPos - A + B;};

		// UV
		if(localVertexIndex == 0u) { vout.uv = vec2f(-1.0f, -1.0f);};
		if(localVertexIndex == 1u) { vout.uv = vec2f( 1.0f, -1.0f);};
		if(localVertexIndex == 2u) { vout.uv = vec2f( 1.0f,  1.0f);};
		
		if(localVertexIndex == 3u) { vout.uv = vec2f(-1.0f, -1.0f);};
		if(localVertexIndex == 4u) { vout.uv = vec2f( 1.0f,  1.0f);};
		if(localVertexIndex == 5u) { vout.uv = vec2f(-1.0f,  1.0f);};

	}
	
	vout.position = uniforms.proj * viewPos;
	vout.color = a_color[splatIndex];

	return vout;
}

@fragment
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var d = length(fragment.uv);

	var fout = FragmentOut();

	var opacity = fragment.color.a * (1.0f - d);
	fout.color = vec4f(fragment.color.xyz * opacity, opacity);

	if(d > 1.0f) {
		discard;
	}

	// if(fout.color.a == 0.0f){
	// 	discard;
	// }
	

	return fout;
}
