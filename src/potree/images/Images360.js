
import {
	SceneNode, Vector3, Matrix4, EventDispatcher, StationaryControls,
	Box3
} from "potree";
import {proj4} from "proj4";

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
	@builtin(vertex_index) vertexID : u32,
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

	var QUAD_POS : array<vec3<f32>, 6> = array<vec3<f32>, 6>(
		vec3<f32>(-1.0, -1.0, 0.0),
		vec3<f32>( 1.0, -1.0, 0.0),
		vec3<f32>( 1.0,  1.0, 0.0),

		vec3<f32>(-1.0, -1.0, 0.0),
		vec3<f32>( 1.0,  1.0, 0.0),
		vec3<f32>(-1.0,  1.0, 0.0),
	);

	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(vertex.position, 1.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;

	var worldSize = 1.0;
	var sizeR = 0.0;
	{
		var viewPosR : vec4<f32> = uniforms.worldView * vec4<f32>(vertex.position, 1.0);
		viewPosR.x = viewPosR.x + 0.5 * worldSize;
		viewPosR.y = viewPosR.y + 0.5 * worldSize;
		var projPosR : vec4<f32> = uniforms.proj * viewPosR;

		var diff = abs((projPosR.x / projPosR.w) - (projPos.x / projPos.w));

		sizeR = 1.0 * uniforms.screen_width * diff;
	}

	let quadVertexIndex : u32 = vertex.vertexID % 6u;
	var pos_quad : vec3<f32> = QUAD_POS[quadVertexIndex];

	var size = max(sizeR, uniforms.size);
	size = min(size, 32.0);

	var fx : f32 = projPos.x / projPos.w;
	fx = fx + size * pos_quad.x / uniforms.screen_width;
	projPos.x = fx * projPos.w;

	var fy : f32 = projPos.y / projPos.w;
	fy = fy + size * pos_quad.y / uniforms.screen_height;
	projPos.y = fy * projPos.w;

	var vout : VertexOut;
	vout.position = projPos;
	vout.pointID = vertex.instanceID;

	return vout;
}

@fragment
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;

	if(i32(fragment.pointID) == uniforms.hoveredIndex){
		fout.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
	}else{
		fout.color = vec4<f32>(
			253.0 / 255.0,
			174.0 / 255.0,
			97.0 / 255.0,
			1.0);
	}

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

export class Image360{

	constructor(){
		this.position = new Vector3();
		this.rotation = new Vector3();
		this.name = "";
	}

}

export class Images360 extends SceneNode{

	constructor(images){
		super(); 

		this.images = images;
		this.uniformBuffer = null;
		this.bindGroup = null;
		this.hoveredIndex = null;
		this.dispatcher = new EventDispatcher();
		this.stationaryControls = new StationaryControls(Potree.instance.renderer.canvas);

		this.positions = new Float32Array(3 * this.images.length);
		for(let i = 0; i < this.images.length; i++){
			let image = this.images[i];

			this.positions[3 * i + 0] = image.position.x;
			this.positions[3 * i + 1] = image.position.y;
			this.positions[3 * i + 2] = image.position.z;
		}

		this.dispatcher.addEventListener("drag", (e) => {
			console.log("drag", e);
		});

		this.dispatcher.addEventListener("click", async (e) => {
			console.log("clicked: ", e.hovered.image.name);

			let image = e.hovered.image;
			let position = this.position.clone().add(image.position);

			this.stationaryControls.pivot.copy(position);
			this.stationaryControls.radius = 0;
			this.stationaryControls.setLabel(`Currently viewing: ${image.name}`);

			Potree.instance.setControls(this.stationaryControls);

			{
				// TODO: prototype hack
				let imageName = image.name.split("\\").at(-1).replaceAll("\"", "");
				let path = `./resources/Drive3/${imageName}`;
				let response = await fetch(path);
				let buffer = await response.arrayBuffer();
				let mimeType = "image/jpg";

				var u8 = new Uint8Array(buffer);
				var blob = new Blob([u8], {type: mimeType});

				let imageBitmap = await createImageBitmap(blob);

				this.stationaryControls.sphereMap.setImage(imageBitmap);

				window.factors = window.factors = [
					-1, -2 * Math.PI / 4, 
					1, 0, 
					1, 0,
				]
				// let [a, b, c, d, e, f] = window.factors ?? [0, 0, 0, 0, 0, 0];

				// let yaw = new Matrix4().rotate(
				// 	a * image.rotation.x - b,
				// 	new Vector3(0, 0, 1));
				// let pitch = new Matrix4().rotate(
				// 	-image.rotation.y,
				// 	new Vector3(1, 0, 0));
				// let roll = new Matrix4().rotate(
				// 	+image.rotation.z - 0 * Math.PI / 4,
				// 	new Vector3(0, 1, 0));
				
				// let radians = (degrees) => Math.PI * degrees / 180;
				// let yaw = new Matrix4().rotate(
				// 	-image.rotation.x - 2 * Math.PI / 4,
				// 	new Vector3(0, 0, 1));
				// let pitch = new Matrix4().rotate(
				// 	-image.rotation.y,
				// 	new Vector3(1, 0, 0));
				// let roll = new Matrix4().rotate(
				// 	+image.rotation.z - 0 * Math.PI / 4,
				// 	new Vector3(0, 1, 0));

				// let rotation = new Matrix4()
				// 	.multiplyMatrices(yaw, pitch);

				// let rotation = new Matrix4().multiply(pitch).multiply(yaw);
				// let rotation = new Matrix4().multiply(roll).multiply(pitch).multiply(yaw);
				// let rotation = new Matrix4().multiply(yaw).multiply(pitch);

				this.stationaryControls.sphereMap.imageRotation = image.rotation;
				// this.stationaryControls.sphereMap.rotation.copy(rotation);

			}

			console.log(position);

		});

	}

	setHovered(index){
		this.hoveredIndex = index;
		this.dispatcher.dispatch("hover", {
			images: this,
			index: index,
			image: this.images[index],
		});
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

	render(drawstate){

		let {renderer} = drawstate;

		init(renderer);

		this.updateUniforms(drawstate);

		let {passEncoder} = drawstate.pass;

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, this.bindGroup);

		let vboPosition = renderer.getGpuBuffer(this.positions);

		passEncoder.setVertexBuffer(0, vboPosition);

		let numVertices = this.positions.length / 3;
		passEncoder.draw(6, numVertices, 0, 0);

		Potree.state.renderedElements += numVertices;
		Potree.state.renderedObjects.push({node: this, numElements: numVertices});

		if(this.isHighlighted){
			let pos = this.boundingBox.center();
			let size = this.boundingBox.size();
			let color = new Vector3(255, 0, 0);

			renderer.drawBoundingBox(pos, size, color,);
		}

	}


}

export class Images360Loader{

	static async load(url, target_crs){

		let response = await fetch(url);
		let text = await response.text();
		let lines = text.split("\n");
		
		let box = new Box3();

		let imageList = [];

		for(let i = 1; i < lines.length; i++){
			let line = lines[i];

			if(line.trim().length === 0){
				continue;
			}

			let tokens = line.split("\t");

			console.log(tokens);


			let imgPath = tokens[0];
			let long = Number(tokens[2]);
			let lat = Number(tokens[3]);
			let z = Number(tokens[4]);

			let [x, y] = proj4(target_crs, [long, lat]);
			box.expandByXYZ(x, y, z);

			let image = new Image360();
			image.position.x = x;
			image.position.y = y;
			image.position.z = z;
			image.name = imgPath;

			imageList.push(image);
		}

		for(let image of imageList){
			image.position.x -= box.min.x;
			image.position.y -= box.min.y;
			image.position.z -= box.min.z;
		}

		let images = new Images360(imageList);
		images.position.copy(box.min);
		images.boundingBox.copy(box);
		images.name = "Helimap MLS Drive 3 Images";
		// scene.add(scene.root, images);

		return images;
	}

}