import Link from "next/link";

export default function Navbar() {
    return (
        <nav className={`w-full bg-white shadow-md p-4 flex items-center dark:text-black transition-all duration-300`}>

            <h1 className="text-xl font-bold">
                MySite
            </h1>

            <div className="flex gap-6 ml-10">
                <Link href="/">Home</Link>
                <Link href="/about">About</Link>
                <Link href="/contact">Contact</Link>
            </div>

            <div className="ml-auto">
                <button className="px-3 py-1 bg-blue-500 text-white rounded">
                    Login
                </button>
            </div>

        </nav>
    );
}