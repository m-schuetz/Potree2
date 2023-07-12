
import {SceneNode, Vector3, Matrix4, EventDispatcher, StationaryControls} from "potree";

let shaderCode = `

struct Uniforms {
	worldView        : mat4x4f,
	proj             : mat4x4f,
	screen_width     : f32,
	screen_height    : f32,
	size             : f32,
	elementCounter   : u32,
	hoveredIndex     : i32,
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexIn{
	@location(0) position : vec3<f32>,
	@builtin(vertex_index) vertex_index : u32,
	@builtin(instance_index) instanceID : u32,
};

struct VertexOut{
	@builtin(position) position : vec4<f32>,
	@location(0) @interpolate(flat) pointID : u32,
};

struct FragmentIn{
	@location(0) @interpolate(flat) pointID : u32,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
	@location(1) point_id : u32,
};

@vertex
fn main_vertex(vertex : VertexIn) -> VertexOut {

	var vout = VertexOut();

	if(vertex.vertex_index == 0u){
		vout.position = vec4<f32>(0.2, 0.2, 0.1, 1.0);
	}else if(vertex.vertex_index == 1u){
		vout.position = vec4<f32>(0.4, 0.2, 0.1, 1.0);
	}else if(vertex.vertex_index == 2u){
		vout.position = vec4<f32>(0.4, 0.4, 0.1, 1.0);
	}else if(vertex.vertex_index == 3u){
		vout.position = vec4<f32>(0.2, 0.2, 0.1, 1.0);
	}else if(vertex.vertex_index == 4u){
		vout.position = vec4<f32>(0.4, 0.4, 0.1, 1.0);
	}else if(vertex.vertex_index == 5u){
		vout.position = vec4<f32>(0.2, 0.4, 0.1, 1.0);
	}

	vout.pointID = vertex.instanceID;

	return vout;
}

@fragment
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout = FragmentOut();
	fout.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
	fout.point_id = uniforms.elementCounter + fragment.pointID;

	return fout;
}

`;

let initialized = false;
let pipeline = null;

function init(renderer){

	if(initialized){
		return;
	}
	
	let {device} = renderer;

	let module = device.createShaderModule({code: shaderCode});

	pipeline = device.createRenderPipeline({
		layout: "auto",
		vertex: {
			module,
			entryPoint: "main_vertex",
			buffers: [
				{ // position
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				}
			]
		},
		fragment: {
			module,
			entryPoint: "main_fragment",
			targets: [
				{format: "bgra8unorm"},
				{format: "r32uint"},
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	initialized = true;
}

export class BVSphere{
	constructor(){
		this.position = new Vector3(0, 0, 0);
		this.radius = 1;
	}
}

export class TDTilesNode{

	constructor(){
		this.boundingVolume = new BVSphere();
		this.children = [];
		this.content = null;
		this.contentLoaded = null;
		this.isLoading = false;
		this.tilesetUri = "";
	}

	traverse(callback){

		callback(this);

		for(let child of this.children){
			child.traverse(callback);
		}

	}

}

export class TDTiles extends SceneNode{

	constructor(url){
		super(); 

		this.url = url;
		this.uniformBuffer = null;
		this.bindGroup = null;
		this.dispatcher = new EventDispatcher();
		this.root = new TDTilesNode();

		this.positions = new Float32Array([
			0.2, 0.2, 0.0,
			0.4, 0.2, 0.0,
			0.4, 0.4, 0.0,
			0.2, 0.2, 0.0,
			0.4, 0.4, 0.0,
			0.2, 0.4, 0.0,
		]);
	}

	setHovered(index){
		// this.hoveredIndex = index;
		// this.dispatcher.dispatch("hover", {
		// 	images: this,
		// 	index: index,
		// 	image: this.images[index],
		// });
	}

	updateUniforms(drawstate){

		let {renderer, camera} = drawstate;
		let {device} = renderer;

		if(this.uniformBuffer === null){
			const uniformBufferSize = 256;

			this.uniformBuffer = device.createBuffer({
				size: uniformBufferSize,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			});

			this.bindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [
					{binding: 0,resource: {buffer: this.uniformBuffer}},
				],
			});
		}

		let data = new ArrayBuffer(256);
		let f32 = new Float32Array(data);
		let view = new DataView(data);

		{ // transform
			// let world = new Matrix4();
			this.updateWorld();
			let world = this.world;
			let view = camera.view;
			let worldView = new Matrix4().multiplyMatrices(view, world);

			f32.set(worldView.elements, 0);
			f32.set(camera.proj.elements, 16);
		}

		{ // misc
			let size = renderer.getSize();

			view.setFloat32(128, size.width, true);
			view.setFloat32(132, size.height, true);
			view.setFloat32(136, 10.0, true);
			view.setUint32(140, Potree.state.renderedElements, true);
			view.setInt32(144, this.hoveredIndex ?? -1, true);
		}

		renderer.device.queue.writeBuffer(this.uniformBuffer, 0, data, 0, data.byteLength);
		
	}

	updateVisibility(renderer, camera){

		let tilesetLoadQueue = [];

		this.root.traverse(node => {

			if(node.content){
				let notLoaded = !node.contentLoaded;
				let isTileset = node.content.uri.endsWith("json");
				if(isTileset && notLoaded){
					tilesetLoadQueue.push(node);
				}
			}

		});


		for(let node of tilesetLoadQueue){

			this.loader.loadNode(node);


			break; // only load one per frame for now
		}


	}

	render(drawstate){

		let {renderer, camera} = drawstate;

		this.updateVisibility(renderer, camera);

		init(renderer);

		this.updateUniforms(drawstate);

		let {passEncoder} = drawstate.pass;

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, this.bindGroup);

		let vboPosition = renderer.getGpuBuffer(this.positions);

		passEncoder.setVertexBuffer(0, vboPosition);

		let numVertices = this.positions.length / 3;
		passEncoder.draw(6, 1, 0, 0);

		Potree.state.renderedElements += numVertices;
		Potree.state.renderedObjects.push({node: this, numElements: numVertices});



		// renderer.drawBoundingBox(
		// 	this.root.boundingVolume.position,
		// 	new Vector3(1, 1, 1).multiplyScalar(this.root.boundingVolume.radius),
		// 	new Vector3(255, 0, 255),
		// );

		this.root.traverse(node => {
			renderer.drawBoundingBox(
				node.boundingVolume.position,
				new Vector3(1, 1, 1).multiplyScalar(node.boundingVolume.radius),
				node.dbgColor,
			);
		});

	}


}