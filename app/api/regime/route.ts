import { exec } from "child_process";
import { NextResponse } from "next/server";

export async function GET() {

  return new Promise((resolve) => {

    exec(
      `"C:\\Users\\k0259\\OneDrive\\Desktop\\Regime_Prediction\\market_regime_radar\\app\\backend\\dist\\run_analysis.exe`,
      (error, stdout, stderr) => {

        if (error) {
          console.error(stderr);

          resolve(
            NextResponse.json(
              {
                error: "Python execution failed",
                err_string: String(error)
               },
              { status: 500 }
            )
          );
          return;
        }

        try {
          const data = JSON.parse(stdout);

          resolve(
            NextResponse.json(data)
          );

        } catch (err) {

          console.error("JSON parse error:", stdout);

          resolve(
            NextResponse.json(
              { error: "Invalid Python output" },
              { status: 500 }
            )
          );
        }
      }
    );

  });
}