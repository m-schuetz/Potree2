
export class Potree3Loader{

	constructor(){

	}

	static async load(url){
		
		let pPositions = fetch(`${url}/voxel_positions.bin`);
		let pColors = fetch(`${url}/voxel_colors.bin`);

		let [rPositions, rColors] = await Promise.all([pPositions, pColors]);

		let bPositions = await rPositions.arrayBuffer();
		let bColors = await rColors.arrayBuffer();

		let positions = new Float32Array(bPositions);
		let colors = new Uint8Array(bColors);

		return {positions, colors};
	}

}