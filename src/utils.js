
function createSvgGradient(scheme){

	// this is what we are creating:
	//
	//<svg width="1em" height="3em"  xmlns="http://www.w3.org/2000/svg">
	//	<defs>
	//		<linearGradient id="gradientID" gradientTransform="rotate(90)">
	//		<stop offset="0%"  stop-color="rgb(93, 78, 162)" />
	//		...
	//		<stop offset="100%"  stop-color="rgb(157, 0, 65)" />
	//		</linearGradient>
	//	</defs>
	//	
	//	<rect width="100%" height="100%" fill="url('#myGradient')" stroke="black" stroke-width="0.1em"/>
	//</svg>

	const gradientId = `${Math.random()}_${Date.now()}`;
	
	const svgn = "http://www.w3.org/2000/svg";
	const svg = document.createElementNS(svgn, "svg");
	svg.setAttributeNS(null, "width", "2em");
	svg.setAttributeNS(null, "height", "3em");
	
	{ // <defs>
		const defs = document.createElementNS(svgn, "defs");
		
		const linearGradient = document.createElementNS(svgn, "linearGradient");
		linearGradient.setAttributeNS(null, "id", gradientId);
		linearGradient.setAttributeNS(null, "gradientTransform", "rotate(90)");

		let n = 32;
		for(let i = 0; i <= n; i++){
			let u = i / n;
			let stopVal = scheme.get(u);

			const percent = u * 100;
			const [r, g, b, a] = stopVal.map(v => parseInt(v));

			const stop = document.createElementNS(svgn, "stop");
			stop.setAttributeNS(null, "offset", `${percent}%`);
			stop.setAttributeNS(null, "stop-color", `rgb(${r}, ${g}, ${b})`);

			linearGradient.appendChild(stop);
		}

		defs.appendChild(linearGradient);
		svg.appendChild(defs);
	}

	const rect = document.createElementNS(svgn, "rect");
	rect.setAttributeNS(null, "width", `100%`);
	rect.setAttributeNS(null, "height", `100%`);
	rect.setAttributeNS(null, "fill", `url("#${gradientId}")`);
	rect.setAttributeNS(null, "stroke", `black`);
	rect.setAttributeNS(null, "stroke-width", `0.1em`);

	svg.appendChild(rect);
	
	return svg;
}

function domFindByName(parent, name){
	let result = Array.from(parent.getElementsByTagName("span")).find(node => node.getAttribute("name") === name);

	return result;
}

export const Utils = {
	createSvgGradient, 
	domFindByName
};