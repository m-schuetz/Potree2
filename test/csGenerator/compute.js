
export function getCsSource(numPoints){
	
	return `
		#version 450

		struct Vertex{
			float x;
			float y;
			float z;
			uint colors;
		};

		layout(std140, set = 0, binding = 0) buffer PositionsOut {
			Vertex points[${numPoints}];
		};

		void main(){
			uint index = gl_GlobalInvocationID.x;

			float size = sqrt(float(${numPoints}));

			float x = 2 * mod(index, size) / size - 1;
			float y = 2 * index / (size * size) - 1;
			float z = 0.3 * cos(3.14 * x) * cos(3.14 * y);

			//float u = float(index) / float(${numPoints});

			Vertex point;
			point.x = x;
			point.y = y;
			point.z = z;

			int r = int(300 * (z + 0.3));

			point.colors = 0xFF000000 | r;

			points[index] = point;
		}

	`
};

export function prepareCsGenerator(renderer, numPoints, targetBuffer){
	let {device} = renderer;

	let shader = renderer.makeShaderModule('compute', getCsSource(numPoints));

	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				type: "storage-buffer"
			}
		]
	});

	const bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{
				binding: 0,
				resource: {
					buffer: targetBuffer,
				}
			},
		]
	});

	const pipeline = device.createComputePipeline({
		layout: device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
		computeStage: {
			module: shader,
			entryPoint: "main"
		}
	});

	return {
		bindGroupLayout: bindGroupLayout,
		bindGroup: bindGroup,
		pipeline: pipeline,
	};
}

