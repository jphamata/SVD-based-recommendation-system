// Deploy the snarkjs-exported Groth16 verifier in an in-memory EVM (ethereumjs)
// and report the gas of verifyProof for the given calldata.
//   node gas.mjs DIR   (DIR/sol/*Groth16Verifier.{bin,abi}, DIR/calldata.txt)
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { EVM } from '@ethereumjs/evm'
import { hexToBytes, bytesToHex } from '@ethereumjs/util'
import { Interface } from 'ethers'

const dir = process.argv[2]
const sol = join(dir, 'sol')
const bin = readFileSync(join(sol, readdirSync(sol).find(f => f.endsWith('Groth16Verifier.bin'))), 'utf8').trim()
const abi = JSON.parse(readFileSync(join(sol, readdirSync(sol).find(f => f.endsWith('Groth16Verifier.abi'))), 'utf8'))
const args = JSON.parse('[' + readFileSync(join(dir, 'calldata.txt'), 'utf8') + ']')
const iface = new Interface(abi)
const evm = await EVM.create()
const gasLimit = 30_000_000n
const dep = await evm.runCall({ data: hexToBytes('0x' + bin), gasLimit })
const to = dep.createdAddress
const call = async a => {
  const data = hexToBytes(iface.encodeFunctionData('verifyProof', a))
  const r = await evm.runCall({ to, data, gasLimit })
  const zeros = data.filter(b => b === 0).length
  const intrinsic = 21000 + 4 * zeros + 16 * (data.length - zeros)
  return { ok: iface.decodeFunctionResult('verifyProof', bytesToHex(r.execResult.returnValue))[0],
           exec: Number(r.execResult.executionGasUsed), total: Number(r.execResult.executionGasUsed) + intrinsic }
}
const good = await call(args)
const bad = args.map(x => x); bad[3] = bad[3].map((v, i) => i === 0 ? '0x' + (BigInt(v) + 1n).toString(16) : v)
const forged = await call(bad)
console.log(JSON.stringify({ deployed: !!to, valid: good, forged: forged.ok, public_inputs: args[3].length }))
